import asyncio
from types import SimpleNamespace
import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from verl import DataProto
from verl.experimental.reward_loop.reward_manager.dapo import DAPORewardManager
async def main():
    config=OmegaConf.create({'reward':{},'trainer':{},'actor_rollout_ref':{'rollout':{'multi_turn':{'turn_level_rewards':False}}}})
    seen=[]
    def score(**kwargs):
        seen.append(kwargs['extra_info'])
        return float(kwargs['extra_info']['num_turns'])
    obj=DAPORewardManager(config,SimpleNamespace(decode=lambda *a,**k:'answer'),score)
    data=DataProto(batch=TensorDict({'responses':torch.tensor([[1,2]]),'attention_mask':torch.tensor([[1,1,1]])},batch_size=[1]),non_tensor_batch={'data_source':np.array(['toy']),'reward_model':np.array([{'ground_truth':'x'}],dtype=object),'extra_info':np.array([{'num_turns':7,'tool_rewards':[123.0]}],dtype=object),'tool_extra_fields':np.array([{'tool_rewards':[2.0]}],dtype=object),'__num_turns__':np.array([3])})
    result=await obj.run_single(data)
    print(result['reward_score'],seen)
asyncio.run(main())
