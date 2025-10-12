import torch

from ComfyUI.folder_paths import get_full_path_or_raise
from ComfyUI.comfy.sd import load_diffusion_model

def load_unet(self, unet_name ="flux1-fill-dev-fp8.safetensors"):
    model_options = {}
    model_options["dtype"] = torch.float8_e4m3fn

    unet_path = get_full_path_or_raise("diffusion_models", unet_name)
    model = load_diffusion_model(unet_path, model_options=model_options)
    return (model,)

if __name__ == "__main__":
    
    load_unet("flux1-fill-dev-fp8.safetensors")
