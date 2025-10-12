import torch
import joblib
from diffusers import FluxFillPipeline, FluxPriorReduxPipeline, AutoencoderKL, FluxTransformer2DModel, FlowMatchEulerDiscreteScheduler
from transformers import T5EncoderModel, CLIPTextModel, CLIPTokenizer, T5TokenizerFast
from diffusers.utils import load_image
from huggingface_hub import hf_hub_download
from torchvision.transforms import ToPILImage

from modules.image_pre import PreprocessImage, PreprocessConfig
from sampling.utils import zero_out, apply_flux_guidance
from sampling.apply_clip import run_clip
from sampling.apply_style import load_style_model, apply_stylemodel, CLIPOutputWrapper
from sampling.config import GenerateConfig


def prepare_embeddings_for_diffusers(positive_conds, negative_conds, dtype, device):
    uncond_prompt_embeds = negative_conds[0][0].to(dtype=dtype, device=device)
    cond_prompt_embeds = positive_conds[0][0].to(dtype=dtype, device=device)
    prompt_embeds = torch.cat([uncond_prompt_embeds, cond_prompt_embeds], dim=0)

    pool_key = 'pooled_output'
    if pool_key not in positive_conds[0][1] or pool_key not in negative_conds[0][1]:
        raise ValueError(f"Could not find key '{pool_key}' in the conditioning dictionary.")

    uncond_pooled_embeds = negative_conds[0][1][pool_key].to(dtype=dtype, device=device)
    cond_pooled_embeds = positive_conds[0][1][pool_key].to(dtype=dtype, device=device)
    pooled_prompt_embeds = torch.cat([uncond_pooled_embeds, cond_pooled_embeds], dim=0)
    
    return prompt_embeds, pooled_prompt_embeds


def generate(subject_url : str, garment_url : str, params : GenerateConfig):
    REPO_FLUX = "black-forest-labs/FLUX.1-Fill-dev"
    REPO_REDUX = hf_hub_download(
        repo_id = "black-forest-labs/FLUX.1-Redux-dev",
        filename= "flux1-redux-dev.safetensors"
    )
    REPO_REDUX_HF = "black-forest-labs/FLUX.1-Redux-dev"

    REPO_ACE = "ali-vilab/ACE_Plus"
    REPO_ACE_SUB = "subject"
    ACE_NAME = "comfyui_subject_lora16.safetensors"
    CONDITIONING_PATH = "/teamspace/studios/this_studio/Redux-Finetune/TryOnPort/text_cache.conditioning"
    REPO_CLIP = "google/siglip-so400m-patch14-384"

    initial_conditioning = None
    conditioning = []
    if params.cache_conditioning:
        initial_conditioning = joblib.load(CONDITIONING_PATH)

        for ts, dict_part in initial_conditioning:
            ts = ts.to(params.device)
            new_dict_part = {k: v.to(params.device) if isinstance(v, torch.Tensor) else v for k, v in dict_part.items()}
            conditioning.append([ts, new_dict_part])

    garment_img = load_image(garment_url)
    clip_vis_tensor = run_clip(garment_img, REPO_CLIP).to(params.device)
    clip_vis_output = CLIPOutputWrapper(clip_vis_tensor)
    style_model = load_style_model(REPO_REDUX)
    style_model.model = style_model.model.to(params.device)
    styled_conds, = apply_stylemodel(conditioning, style_model, clip_vis_output, params.redux_strength, params.redux_strength_type)

    negative_conds, = zero_out(styled_conds)
    positive_conds, = apply_flux_guidance(styled_conds, params.flux_guidance)
    
    prompt_embeds, pooled_prompt_embeds = prepare_embeddings_for_diffusers(
        positive_conds, negative_conds, params.dtype, params.device
    )
    print(f"Custom Embeddings : {prompt_embeds.size()}")

    # main pipe starts
    pipe_redux = FluxPriorReduxPipeline.from_pretrained(REPO_REDUX_HF, torch_dtype = params.dtype).to(params.device)
    pipe_prior_output = pipe_redux(
        garment_img,
        prompt_embeds= prompt_embeds,
        pooled_prompt_embeds= pooled_prompt_embeds

    )

    prompt_embeds = pipe_prior_output.prompt_embeds
    pooled_prompt_embeds = pipe_prior_output.pooled_prompt_embeds
    
    # NOW apply your redux_strength and redux_strength_type adjustments to the embeddings
    # if params.redux_strength != 1.0:
    #     if params.redux_strength_type == "multiply":
    #         prompt_embeds = old_prompt_embeds * params.redux_strength
    #         pooled_prompt_embeds = pooled_prompt_embeds * params.redux_strength
    #     elif params.redux_strength_type == "add":
    #         prompt_embeds = prompt_embeds + params.redux_strength
    #         pooled_prompt_embeds = pooled_prompt_embeds + params.redux_strength
    
    # print(torch.allclose(prompt_embeds, old_prompt_embeds))

    
    print(f"Flux Embeddings Format :{pipe_prior_output.prompt_embeds.size()}")
    pipe = FluxFillPipeline.from_pretrained(REPO_FLUX, torch_dtype = params.dtype).to(params.device)
    # pipe.load_lora_weights(REPO_ACE, subfolder = REPO_ACE_SUB, filename = ACE_NAME)
    processor_config = PreprocessConfig(
        resized_height= 1024,
    )
    processor = PreprocessImage(processor_config)
    image, mask = processor.preprocess(subject_url, garment_url)

    pil_image = ToPILImage()(image[0])
    pil_mask = ToPILImage()(mask[0])
    pil_image.save('subject_image.png')
    pil_mask.save('mask_image.png')

    image = image.to(device = params.device, dtype = params.dtype)
    mask = mask.to(device = params.device, dtype = params.dtype)

    out = pipe(
        image = image,
        mask_image = mask,
        # prompt_embeds= prompt_embeds,
        # pooled_prompt_embeds= pooled_prompt_embeds,
        guidance_scale= 40,
        num_inference_steps= params.num_steps,
        strength = 1.,
        generator = torch.Generator(params.device).manual_seed(params.seed),
        # joint_attention_kwargs= {"scale" : params.ACE_scale},
        **pipe_prior_output
    ).images[0]

    out.save('output_fill_1.png')
    print("Saved Output")

if __name__ == "__main__":
    params = GenerateConfig(
        num_steps= 30,
        seed = 42,
        sampler = 'euler',
        scheduler= 'simple',
        flux_guidance = 50,
        CFG = 0.8,
        redux_strength= 1.50,
        redux_strength_type= "multiply",
        ACE_scale= 0.,
        dtype = torch.bfloat16,
    )

    subject_url = "https://res.cloudinary.com/dukgi26uv/image/upload/v1759842454/the-nude-v-neck-pointelle-knit-tee-tops-snhkxv_2048x_bfnch4.webp"
    garment_url = "https://res.cloudinary.com/dukgi26uv/image/upload/v1759842480/Manchester_United_Home_92.94_Shirt_kyajol.webp"

    generate(subject_url, garment_url, params)









