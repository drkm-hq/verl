import asyncio
from types import SimpleNamespace
import numpy as np
import torch
from tensordict import TensorDict
from omegaconf import OmegaConf
from verl import DataProto
from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager
from verl.experimental.reward_loop.reward_manager.dapo import DAPORewardManager
from verl.experimental.reward_loop.reward_manager.gdpo import GDPORewardManager

async def main():
    config=OmegaConf.create({'reward':{},'trainer':{'experiment_name':'p4-check'}})
    for manager in (NaiveRewardManager,DAPORewardManager,GDPORewardManager):
        captured={}
        async def score(**kwargs):
            captured.update(kwargs['extra_info'])
            return float(kwargs['extra_info'].get('episode_done',False))
        obj=manager(config,SimpleNamespace(decode=lambda *a,**k:'final response'),score)
        data=DataProto(batch=TensorDict({'responses':torch.tensor([[1,2]]),'attention_mask':torch.tensor([[1,1,1]])},batch_size=[1]),non_tensor_batch={'data_source':np.array(['toy']), 'reward_model':np.array([{'ground_truth':'x'}],dtype=object), 'extra_info':np.array([{}],dtype=object), 'tool_extra_fields':np.array([{'episode_done':True,'turn_rewards':[2.0],'generated_turn_spans':[(0,2)]}],dtype=object)})
        result=await obj.run_single(data)
        print(manager.__name__, 'callback extra_info=',captured,'reward=',result['reward_score'])
asyncio.run(main())
