from pydantic import BaseModel
import torch
from typing import List, Union, Literal

class GenerateConfig(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    num_steps : int
    seed : int
    sampler : Union[Literal['euler'], Literal['dmpp_3_sde']]
    scheduler : Union[Literal['simple'], Literal['kl_optimal']]
    flux_guidance : float
    cache_conditioning : bool = True
    CFG : float
    device : str = 'cuda'
    dtype : torch.dtype = torch.bfloat16
    redux_strength : float
    redux_strength_type : Union[Literal['multiply'], Literal['attn_bias']]
    ACE_scale : float