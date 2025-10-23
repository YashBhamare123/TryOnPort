import torch
import torch.nn.functional as F
import time
import joblib
from diffusers import FluxFillPipeline, FluxPriorReduxPipeline, AutoencoderKL, FluxTransformer2DModel, FlowMatchEulerDiscreteScheduler
from transformers import T5EncoderModel, CLIPTextModel, CLIPTokenizer, T5Tokenizer
from diffusers.utils import load_image
from huggingface_hub import hf_hub_download
from torchvision.transforms import ToPILImage, ToTensor
import torch.nn as nn
from concurrent.futures import ProcessPoolExecutor
import torch.multiprocessing

from modules.image_pre import PreprocessImage, PreprocessConfig
from sampling.utils import zero_out, apply_flux_guidance
from sampling.apply_clip import run_clip
from sampling.apply_style import load_style_model, apply_stylemodel, CLIPOutputWrapper
from sampling.config import GenerateConfig
from Logo.detector import process_logo, deconcatenation
from sampling.teacache import teacache_forward


def _load_components(input : dict):
    pipeline = input['pipeline']
    repo = input['repo']
    subfolder = input['subfolder']
    device = input['device']
    dtype = input['dtype']
    dtype_kwargs = {}
    if dtype:
        dtype_kwargs['torch_dtype'] = dtype

    if subfolder in ['transformer', 'vae']:

        if device:
            model = pipeline.from_pretrained(repo, subfolder = subfolder, low_cpu_mem_usage=False, local_files_only=True, **dtype_kwargs).to(device)
        else:
            model = pipeline.from_pretrained(repo, subfolder = subfolder,low_cpu_mem_usage=False, local_files_only=True, **dtype_kwargs)
        return {subfolder : model}
    
    else:
        return {subfolder : None}
def load_pipe_flux( device : str, dtype : torch.dtype) -> FluxFillPipeline:
    components = [
    {
        'pipeline' : CLIPTokenizer,
        'repo' : 'black-forest-labs/FLUX.1-Fill-dev',
        'subfolder' : 'tokenizer',
        'device' : None,
        'dtype' : None
    },
    {
        'pipeline' : T5Tokenizer,
        'repo' : 'black-forest-labs/FLUX.1-Fill-dev',
        'subfolder' : 'tokenizer_2',
        'device' : None,
        'dtype' : None
    },
    {
        'pipeline' : AutoencoderKL,
        'repo' : 'black-forest-labs/FLUX.1-Fill-dev',
        'subfolder' : 'vae',
        'device' : device,
        'dtype' : dtype
    },
    {
        'pipeline' : CLIPTextModel,
        'repo' : 'black-forest-labs/FLUX.1-Fill-dev',
        'subfolder' : 'text_encoder',
        'device' : device,
        'dtype' :dtype
    },
    {
        'pipeline' : T5EncoderModel,
        'repo' : 'black-forest-labs/FLUX.1-Fill-dev',
        'subfolder' : 'text_encoder_2',
        'device' : device,
        'dtype' : dtype
    },
    {
        'pipeline' : FluxTransformer2DModel,
        'repo' : 'black-forest-labs/FLUX.1-Fill-dev',
        'subfolder' : 'transformer',
        'device' : device,
        'dtype' : dtype
    }
]
    out = {}
    for component in components:
        out.update(_load_components(component))
    scheduler = FlowMatchEulerDiscreteScheduler() 
    out.update({"scheduler" : scheduler})

    pipe = FluxFillPipeline(**out).to(device)
    return pipe
    


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
        low_cpu_mem_usage=True,
        # do_rescale = False
        ).to(params.device)


    start = time.perf_counter()
    pipe = load_pipe_flux(device = params.device, dtype = params.dtype)
    end = time.perf_counter()

    FluxTransformer2DModel.forward = teacache_forward
    pipe.transformer.__class__.enable_teacache = True
    pipe.transformer.__class__.cnt = 0
    pipe.transformer.__class__.num_steps = params.num_steps
    pipe.transformer.__class__.rel_l1_thresh = 0.4 # 0.25 for 1.5x speedup, 0.4 for 1.8x speedup, 0.6 for 2.0x speedup, 0.8 for 2.25x speedup
    pipe.transformer.__class__.accumulated_rel_l1_distance = 0
    pipe.transformer.__class__.previous_modulated_input = None
    pipe.transformer.__class__.previous_residual = None

    pipe.load_lora_weights(REPO_ACE, subfolder = REPO_ACE_SUB, weight_name = ACE_NAME)

    if params.compile_repeated:
        pipe.transformer.single_transformer_blocks = nn.ModuleList([
            torch.compile(block) for block in pipe.transformer.single_transformer_blocks
        ])

        pipe.transformer.transformer_blocks = nn.ModuleList([
            torch.compile(block) for block in pipe.transformer.transformer_blocks
        ])

    print(f"Pipeline Loading Time {end - start:.2f}")

    pipe_prior_output = pipe_redux(
        garment_img,
        prompt_embeds_scale= params.redux_strength
    )
    
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
        joint_attention_kwargs= {"scale" : 0},
        cfg = params.CFG,
        **pipe_prior_output,
    ).images[0]

    out.save('subject_image.png')   
    out = ToTensor()(out)
    gen_img, gar_img = processor.postprocess(out, subject_width)
    print("gen_img_shape:",gen_img.shape)
    print("gar_img.shape",gar_img.shape)
    print(gar_img[0].shape)

    pil_img = ToPILImage()(gen_img[0])
    pil_gar = ToPILImage()(gar_img[0])
    pil_img.save('subject_image.png')
    pil_gar.save('mask_image.png')

    # print(gen_img.size())
    # print(gar_img.size())

    pixel_space, o_w, inpaint_mask, mm_bbox, logo_images = process_logo(gar_img, gen_img)
    print("pixel_space_dtype:",pixel_space.dtype)
    print("inpaint_mask_type:",inpaint_mask.dtype)

    list_size = logo_images.size()[0]
    logo_redux = []
    logo_embeds = []
    for i in range(list_size):
        img = ToPILImage()(logo_images[i].to(torch.float32))
        pipe_prior_logos = pipe_redux(img)
        logo_redux.append(pipe_prior_logos.prompt_embeds.squeeze(0))
        logo_embeds.append(pipe_prior_logos.pooled_prompt_embeds.squeeze(0))

    pooled_prompt_embeds = torch.stack(logo_embeds, dim= 0)
    prompt_embeds = torch.stack(logo_redux, dim = 0)
    print(prompt_embeds.size())
    print(pooled_prompt_embeds.size())

    prompt_embeds = prompt_embeds.to(device=params.device, dtype=params.dtype)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device=params.device, dtype=params.dtype)

    pixel_space_pil = ToPILImage()(pixel_space[0].to(torch.float32))
    inpaint_mask_pil = ToPILImage()(inpaint_mask[0].to(torch.float32))
    pixel_space_pil.save('subject_image.png')
    inpaint_mask_pil.save('mask_image.png')

    #pixel_space = pixel_space.to(device = params.device, dtype = params.dtype) / 255.
    #inpaint_mask= inpaint_mask.to(device = params.device, dtype= params.dtype)
    
    new_out = pipe(
        image = pixel_space,
        mask_image = inpaint_mask,
        guidance_scale= params.flux_guidance,
        num_inference_steps= params.num_steps,
        strength = 1.,
        generator = torch.Generator(params.device).manual_seed(params.seed),
        joint_attention_kwargs= {"scale" : params.ACE_scale},
        cfg = params.CFG,
        prompt_embeds= prompt_embeds,
        pooled_prompt_embeds= pooled_prompt_embeds,
    ).images

    new_list = [ToTensor()(t) for t in new_out]
    new_ts = torch.stack(new_list, dim = 0)
    print("new_ts_type:",type(new_ts))
    print("new_ts_dtype:",new_ts.dtype)
    print("new_ts_shape:",new_ts.shape)
    for i, img in enumerate(new_ts):
        ToPILImage()(img.to(torch.float32)).save(f"new_ts_{i}.png")

    new_ts = F.interpolate(
        new_ts,
        mode = 'bilinear',
        size = pixel_space.size()[2:],
        align_corners= False
    )

    assert new_ts.size() == pixel_space.size(), f"{new_ts.size()} and {pixel_space.size()} do not match"

    pil_gen = ToPILImage()(gen_img[0])
    out = deconcatenation(gen_img, new_ts, o_w, mm_bbox, blend_amount= 0.0)
    out = ToPILImage()(out[0].to(torch.float32))
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









