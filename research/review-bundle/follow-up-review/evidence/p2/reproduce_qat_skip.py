"""Check P2's new guards against verl's built-in W4A4 activation observer.

CPU-only: loads the unmodified QATMode/QATLinear class definitions from source,
without importing the Triton kernels, and replaces only the CUDA quantizer with
a spy. This proves the accepted train/eval state/quantizer-input difference,
not an end-to-end GPU log-probability difference.
"""
import ast
import tempfile
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from transformers import Qwen3Config

from verl.utils.config import _validate_skip_old_log_prob_config
from verl.utils.model import find_train_mode_dependent_modules
from verl.workers.engine_workers import ActorRolloutRefWorker

SOURCE_ROOT = Path('<repo>/wt/p2')
source = SOURCE_ROOT / 'verl/utils/qat/linear.py'
tree = ast.parse(source.read_text())
nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in {'QATMode', 'QATLinear'}]
assert len(nodes) == 2

class QuantizerSpy:
    seen_amax = []

    @staticmethod
    def apply(x, global_amax, group_size):
        QuantizerSpy.seen_amax.append(global_amax.detach().clone())
        return x

scope = dict(torch=torch, nn=nn, F=F, Enum=Enum, Optional=Optional,
             FP4_E2M1_MAX=6.0, FP8_E4M3_MAX=448.0, STEFP4QuantTriton=QuantizerSpy)
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), scope)

with tempfile.TemporaryDirectory() as tmp:
    Qwen3Config(attention_dropout=0.0).save_pretrained(tmp)
    with initialize_config_dir(version_base=None, config_dir=str(SOURCE_ROOT/'verl/trainer/config')):
        cfg = compose(config_name='ppo_trainer', overrides=[
            'actor_rollout_ref.actor.skip_old_log_prob_when_on_policy=true',
            'data.train_batch_size=8', 'actor_rollout_ref.actor.ppo_mini_batch_size=8',
            'algorithm.adv_estimator=grpo',
            'actor_rollout_ref.actor.fsdp_config.qat.enable=true',
            'actor_rollout_ref.actor.fsdp_config.qat.mode=w4a4',
        ])
    cfg.actor_rollout_ref.model.path = tmp
    _validate_skip_old_log_prob_config(cfg)
    print('W4A4 + skip configuration: ACCEPTED')
    module = scope['QATLinear'](16, 16, mode=scope['QATMode'].W4A4,
                                activation_observer='static_minmax')
    module.input_amax.fill_(1.0)
    module.input_global_scale.fill_(6.0 * 448.0)
    print('Module walk:', find_train_mode_dependent_modules(module))
    worker = SimpleNamespace(config=SimpleNamespace(actor=cfg.actor_rollout_ref.actor),
                             actor=SimpleNamespace(engine=SimpleNamespace(module=module)))
    ActorRolloutRefWorker._check_actor_for_on_policy_skip(worker)
    print('Worker init check: ACCEPTED')
    x = torch.full((1, 16), 2.0)
    module.eval()
    module._fake_quantize_activation(x)
    eval_scale = module.input_global_scale.item()
    module.train()
    module._fake_quantize_activation(x)
    train_scale = module.input_global_scale.item()
    print('Scale at identical weights/input: eval=', eval_scale, 'train=', train_scale)
    print('Actual observer supplied quantizer amax:', [a.item() for a in QuantizerSpy.seen_amax])
    assert eval_scale != train_scale
    assert not torch.equal(QuantizerSpy.seen_amax[0], QuantizerSpy.seen_amax[1])
