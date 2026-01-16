import requests
import io
import torch
import torch.nn.functional as F
import time
import joblib
from diffusers import FluxFillPipeline, FluxPriorReduxPipeline, AutoencoderKL, FluxTransformer2DModel, FlowMatchEulerDiscreteScheduler, FluxControlNetModel, FluxFillControlNetPipeline
from transformers import T5EncoderModel, CLIPTextModel, CLIPTokenizer, T5Tokenizer
from diffusers.utils import load_image
from huggingface_hub import hf_hub_download
from torchvision.transforms import ToPILImage, ToTensor
import torch.nn as nn
from PIL import Image

from modules.image_pre import PreprocessImage, PreprocessConfig, resize_image
from sampling.config import GenerateConfig
from logo.detector import process_logo, deconcatenation
from sampling.teacache import teacache_forward


class TryOnPipeline:
    REPO_REDUX_HF = "black-forest-labs/FLUX.1-Redux-dev"
    REPO_ACE = "ali-vilab/ACE_Plus"
    REPO_ACE_SUB = "subject"
    ACE_NAME = "comfyui_subject_lora16.safetensors"
    CONTROL_REPO = "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro"

    def __init__(self, params : GenerateConfig):
        self.params = params
        self.controlnet = FluxControlNetModel.from_pretrained(self.CONTROL_REPO, torch_dtype=torch.bfloat16)
        self.flux_pipe = self.load_flux_with_modules(self._load_pipe_flux(device = self.params.device, dtype= self.params.dtype), self.params)
        self.redux_pipe = FluxPriorReduxPipeline.from_pretrained(
            self.REPO_REDUX_HF,
            torch_dtype = self.params.dtype, 
            low_cpu_mem_usage=True,
            # do_rescale = False
            ).to(self.params.device)
        
        self.processor_config = PreprocessConfig(
            resized_height= self.params.image_res,
            grow_padding= self.params.grow_padding
        )
        self.processor = PreprocessImage(self.processor_config)

    def _load_components(self, input : dict):
        pipeline = input['pipeline']
        repo = input['repo']
        subfolder = input['subfolder']
        device = input['device']
        dtype = input['dtype']
        dtype_kwargs = {}
        if dtype:
            dtype_kwargs['torch_dtype'] = dtype

        # if subfolder in ['transformer', 'vae']:

        if device:
            model = pipeline.from_pretrained(repo, subfolder = subfolder, low_cpu_mem_usage=True, device_map = 'auto', local_files_only=True, **dtype_kwargs).to(device)
        else:
            model = pipeline.from_pretrained(repo, subfolder = subfolder,low_cpu_mem_usage=True,  device_map = 'auto', local_files_only=True, **dtype_kwargs)
        return {subfolder : model}
        
        # else:
            # return {subfolder : None}
    
    def _load_pipe_flux(self, device : str, dtype : torch.dtype) -> FluxFillPipeline:
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
            out.update(self._load_components(component))
        scheduler = FlowMatchEulerDiscreteScheduler() 
        out.update({"scheduler" : scheduler})

        pipe = FluxFillControlNetPipeline(controlnet = self.controlnet, **out).to(device)
        return pipe
    
    def load_flux_with_modules(self, pipe : FluxFillPipeline, params : GenerateConfig):
        # FluxTransformer2DModel.forward = teacache_forward
        # pipe.transformer.__class__.enable_teacache = True
        # pipe.transformer.__class__.cnt = 0
        # pipe.transformer.__class__.num_steps = params.num_steps
        # pipe.transformer.__class__.rel_l1_thresh = params.teacache_coeff # 0.25 for 1.5x speedup, 0.4 for 1.8x speedup, 0.6 for 2.0x speedup, 0.8 for 2.25x speedup
        # pipe.transformer.__class__.accumulated_rel_l1_distance = 0
        # pipe.transformer.__class__.previous_modulated_input = None
        # pipe.transformer.__class__.previous_residual = None

        pipe.load_lora_weights(self.REPO_ACE, subfolder = self.REPO_ACE_SUB, weight_name = self.ACE_NAME)

        if params.compile_repeated:
            pipe.transformer.single_transformer_blocks = nn.ModuleList([
                torch.compile(block) for block in pipe.transformer.single_transformer_blocks
            ])

            pipe.transformer.transformer_blocks = nn.ModuleList([
                torch.compile(block) for block in pipe.transformer.transformer_blocks
            ])
        return pipe
    
    def _make_pil(self, input : torch.Tensor, index : int = 0):
        out = ToPILImage()(input[index].to(dtype= torch.float32))
        return out


    def _make_tensor(self, input):
        img = ToTensor()(input)
        return img.unsqueeze(0)
    
    def _composite_mask(self, image, mask, composite_factor : float = 0.3):
        image = image.to(device = 'cuda')
        mask= mask.to(device = 'cuda')

        background = (1 - mask) * image
        image = image * mask + background * composite_factor
        image = self._make_pil(image)
        return image
    
    def _load_image(self, image_url : str, is_mask : bool = False, local_file = False):

        if local_file:
            image = Image.open(image_url)
        else:
            response = requests.get(image_url)
            image = Image.open(io.BytesIO(response.content))

        if is_mask:
            image = image.convert(mode='L')
        return image


    def __call__(self, subject_url : str, garment_url : str, control_url : str, local_file = False):
        image, mask, subject_width = self.processor.preprocess(subject_url, garment_url, local_file)
        sub_img, gar_img = self.processor.split(image, subject_width)
        dummy = torch.zeros((gar_img.size()[0], 3, gar_img.size()[2], gar_img.size()[3]))
        garment_pil = self._make_pil(gar_img)
       
        redux_embeds = self.redux_pipe(
            garment_pil,
            prompt_embeds_scale= self.params.redux_strength
        )

        print(f"Redux Embedding Size {redux_embeds['prompt_embeds'].size()}")
        # Control image processing to add garment side by side
        control_image = self._load_image(control_url, local_file = local_file)
        control_image = self._make_tensor(control_image)
        control_image = F.interpolate(
            control_image,
            size = (sub_img.size()[2], sub_img.size()[3]),
            mode = 'bilinear',
            align_corners= False
        )
        control_image = torch.concat([control_image, dummy], dim = 3)
        control_image = self._make_pil(control_image)
        

        # Main pipeline starts here
        _, _, H, W = image.size()
        image = image.to(device= self.params.device, dtype = self.params.dtype)
        mask = mask.to(device= self.params.device, dtype = self.params.dtype)

        out = self.flux_pipe(
            # prompt = 'a machester united jersey which is worn by a girl',
            control_image = control_image,
            image = image,
            mask_image = mask,
            height = H,
            width = W,
            guidance_scale = self.params.flux_guidance,
            num_inference_steps = self.params.num_steps,
            joint_attention_kwargs= {"scale" : 0},
            control_mode = 4,
            controlnet_conditioning_scale = 0.9,
            control_guidance_end=0.65,
            # cfg = self.params.CFG,
            generator = torch.Generator(self.params.device).manual_seed(self.params.seed),
            **redux_embeds
        ).images[0]
        return out
        
        out = self._make_tensor(out)
        out = F.interpolate(
            input = out,
            size = (H, W),
            mode = 'bilinear',
            align_corners= False
        )
        gen_img, _ = self.processor.split(out, subject_width)
        mask_d, _ = self.processor.split(mask, subject_width)
        mask_d_comp = self._composite_mask(gen_img, mask_d)

        image, o_w, mask, mm_bbox, logo_images = process_logo(gar_img, gen_img)
        num_logos = logo_images.size()[0]

        if num_logos:
            redux_embeds_logo_prompt = []
            redux_embeds_logo_pooled = []
            for i in range(num_logos):
                logo = self._make_pil(logo_images[i].unsqueeze(0))
                logo_embed = self.redux_pipe(
                    logo,
                    prompt_embeds_scale= self.params.logo_redux_strength
                )
                redux_embeds_logo_prompt.append(logo_embed['prompt_embeds'][0])
                redux_embeds_logo_pooled.append(logo_embed['pooled_prompt_embeds'][0])
            
            redux_embeds_logo = {
                'prompt_embeds' : torch.stack(redux_embeds_logo_prompt, dim = 0),
                'pooled_prompt_embeds' : torch.stack(redux_embeds_logo_pooled, dim = 0),
            }
            _, _, H, W = image.size()

            image = image.to(dtype= self.params.dtype, device= self.params.device)
            mask = mask.to(dtype= self.params.dtype, device = self.params.device)

            out = self.flux_pipe(
                image = image,
                mask_image = mask,
                height = H,
                width = W,
                guidance_scale = self.params.flux_guidance,
                num_inference_steps = self.params.num_steps_logo,
                joint_attention_kwargs= {"scale" : self.params.ACE_scale},
                cfg = self.params.CFG,
                generator = torch.Generator(self.params.device).manual_seed(self.params.seed),
                **redux_embeds_logo
            ).images

            out = torch.stack([self._make_tensor(logo)[0] for logo in out], dim = 0)
            out = F.interpolate(
                input = out,
                size = (H, W),
                mode = 'bilinear',
                align_corners= False
            )

            gen_img = deconcatenation(gen_img, out, o_w, mm_bbox, blend_amount= 0.0)
        
        gen_img = self._make_pil(gen_img)
        return gen_img
    

if __name__ == "__main__":
    params = GenerateConfig(
        num_steps= 5,
        seed = 42,
        sampler = 'euler',
        scheduler= 'simple',
        flux_guidance = 30,
        CFG = 1.,
        redux_strength= 0.7,
        redux_strength_type= "multiply",
        ACE_scale= 1.,
        dtype = torch.bfloat16,
    )

    subject_url = "https://res.cloudinary.com/dukgi26uv/image/upload/v1754051392/tryon-images/r1srtslvfaya3hnpndoh.jpg"
    garment_url = "https://res.cloudinary.com/dukgi26uv/image/upload/v1754049604/tryon-images/vwckkvaqvmxd0ap9xroi.jpg"

    generate(subject_url, garment_url, params)









