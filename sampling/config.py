from pydantic import BaseModel
import torch
from typing import List, Union, Literal

class GenerateConfig(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    num_steps : int
    num_steps_logo : int
    seed : int
    sampler : Union[Literal['euler'], Literal['dmpp_3_sde']]
    flux_guidance : float
    cache_conditioning : bool = True
    CFG : float
    device : str = 'cuda'
    dtype : torch.dtype = torch.bfloat16
    redux_strength : float
    logo_redux_strength : float
    ACE_scale : float
    compile_repeated : bool = False
    teacache_coeff : float = 0.1
    image_res : int = 1024
    grow_padding : int = 20