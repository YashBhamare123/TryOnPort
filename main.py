import torch
import joblib
from diffusers import FluxFillPipeline, FluxPriorReduxPipeline, AutoencoderKL, FluxTransformer2DModel, FlowMatchEulerDiscreteScheduler
from transformers import T5EncoderModel, CLIPTextModel, CLIPTokenizer, T5TokenizerFast
from diffusers.utils import load_image
from huggingface_hub import hf_hub_download
from torchvision.transforms import ToPILImage, ToTensor

from modules.image_pre import PreprocessImage, PreprocessConfig
from sampling.utils import zero_out, apply_flux_guidance
from sampling.apply_clip import run_clip
from sampling.apply_style import load_style_model, apply_stylemodel, CLIPOutputWrapper
from sampling.config import GenerateConfig
from Logo.detector import process_logo, deconcatenation

def generate(subject_url : str, garment_url : str, params : GenerateConfig):

    test = False
    REPO_FLUX = "black-forest-labs/FLUX.1-Fill-dev"
    REPO_REDUX_HF = "black-forest-labs/FLUX.1-Redux-dev"

    REPO_ACE = "ali-vilab/ACE_Plus"
    REPO_ACE_SUB = "subject"
    ACE_NAME = "comfyui_subject_lora16.safetensors"
    CONDITIONING_PATH = "/teamspace/studios/this_studio/Redux-Finetune/TryOnPort/text_cache.conditioning"
    REPO_CLIP = "google/siglip-so400m-patch14-384"

    garment_img = load_image(garment_url)
    pipe_redux = FluxPriorReduxPipeline.from_pretrained(
        REPO_REDUX_HF,
        torch_dtype = params.dtype, 
        low_cpu_mem_usage=True
        ).to(params.device)

    sampler = FlowMatchEulerDiscreteScheduler()
    pipe = FluxFillPipeline.from_pretrained(
        REPO_FLUX,
        torch_dtype = params.dtype,
        low_cpu_mem_usage=True
        ).to(params.device)

    pipe_prior_output = pipe_redux(
        garment_img,
        prompt_embeds_scale= params.redux_strength
    )
    
    pipe.scheduler = sampler
    # pipe.load_lora_weights(REPO_ACE, subfolder = REPO_ACE_SUB, filename = ACE_NAME)
    if test:
        processor_config = PreprocessConfig(
            resized_height= 1024,
        )
        processor = PreprocessImage(processor_config)
        image, mask, subject_width = processor.preprocess(subject_url, garment_url)

        pil_image = ToPILImage()(image[0])
        pil_mask = ToPILImage()(mask[0])
        pil_image.save('subject_image.png')
        pil_mask.save('mask_image.png')
        W, H = pil_image.size
        image = image.to(device = params.device, dtype = params.dtype)
        mask = mask.to(device = params.device, dtype = params.dtype)

        out = pipe(
            image = image,
            mask_image = mask,
            guidance_scale= params.flux_guidance,
            num_inference_steps= params.num_steps,
            strength = 1.,
            generator = torch.Generator(params.device).manual_seed(params.seed),
            height = H,
            width = W,
            # joint_attention_kwargs= {"scale" : params.ACE_scale},
            cfg = params.CFG,
            **pipe_prior_output,
        ).images[0]

        
        # Logo Pipeline
        out = ToTensor()(out)
        img, gar = processor.postprocess(out, subject_width)
        pil_img = ToPILImage()(img[0])
        pil_gar = ToPILImage()(gar[0])
        pil_img.save('subject_image.png')
        pil_gar.save('mask_image.png')

    gen_img = load_image('https://res.cloudinary.com/dukgi26uv/image/upload/v1760462866/output_0_ldopjx.png')
    gar_img = load_image('https://res.cloudinary.com/dukgi26uv/image/upload/v1760462866/output_1_bjkvdg.png')
    
    gen_img = ToTensor()(gen_img)
    gar_img = ToTensor()(gar_img)

    print(gen_img.size())
    print(gar_img.size())

    pixel_space, o_w, inpaint_mask, mm_bbox, logo_images = process_logo(gen_img, gar_img)

    list_size = logo_images.size()[0]
    logo_redux = []
    logo_embeds = []
    for i in range(list_size):
        pipe_prior_logos = pipe_redux(logo_images[i].unsqueeze(0))
        logo_redux.append(pipe_prior_logos.prompt_embeds.squeeze(0))
        logo_embeds.append(pipe_prior_logos.pooled_prompt_embeds.squeeze(0))

    pooled_prompt_embeds = torch.stack(logo_embeds, dim= 0)
    prompt_embeds = torch.stack(logo_redux, dim = 0)
    print(prompt_embeds.size())
    print(pooled_prompt_embeds.size())

    new_out = pipe(
        image = pixel_space,
        mask_image = inpaint_mask,
        guidance_scale= params.flux_guidance,
        num_inference_steps= 3,
        strength = 1.,
        generator = torch.Generator(params.device).manual_seed(params.seed),
        # joint_attention_kwargs= {"scale" : params.ACE_scale},
        cfg = params.CFG,
        prompt_embeds= prompt_embeds,
        pooled_prompt_embeds= pooled_prompt_embeds
    ).images

    new_list = [ToTensor()(t) for t in new_out]
    new_ts = torch.stack(new_list, dim = 0)
    out = deconcatenation(gen_img, new_ts, o_w, mm_bbox)
    out = ToPILImage()(out[0])
    out.save('output_fill_1.png')
    print("Saved Output")

if __name__ == "__main__":
    params = GenerateConfig(
        num_steps= 5,
        seed = 42,
        sampler = 'euler',
        scheduler= 'simple',
        flux_guidance = 30,
        CFG = 1.,
        redux_strength= 0.4,
        redux_strength_type= "multiply",
        ACE_scale= 0.,
        dtype = torch.bfloat16,
    )

    subject_url = "https://res.cloudinary.com/dukgi26uv/image/upload/v1754142589/tryon-images/o5s6nngrevxa05iemley.jpg"
    garment_url = "https://res.cloudinary.com/dukgi26uv/image/upload/v1754141234/tryon-images/wrcx1xhsyvm2bad017h2.webp"

    generate(subject_url, garment_url, params)









