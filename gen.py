import modal

app = modal.App("tryon-inference")

volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
files_volume = modal.Volume.from_name("tryon-files", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "accelerate==1.10.1",
        "annotated-types==0.7.0",
        "anyio==4.11.0",
        "certifi==2025.10.5",
        "charset-normalizer==3.4.3",
        "contourpy==1.3.3",
        "cycler==0.12.1",
        "distro==1.9.0",
        "dotenv==0.9.9",
        "filelock==3.20.0",
        "fonttools==4.60.1",
        "fsspec==2025.9.0",
        "groq==0.32.0",
        "h11==0.16.0",
        "hf-xet==1.1.10",
        "httpcore==1.0.9",
        "httpx==0.28.1",
        "huggingface-hub==0.35.3",
        "idna==3.10",
        "importlib-metadata==8.7.0",
        "Jinja2==3.1.6",
        "joblib==1.5.2",
        "kiwisolver==1.4.9",
        "MarkupSafe==3.0.3",
        "matplotlib==3.10.7",
        "mpmath==1.3.0",
        "networkx==3.5",
        "numpy==2.3.3",
        "nvidia-cublas-cu12==12.8.4.1",
        "nvidia-cuda-cupti-cu12==12.8.90",
        "nvidia-cuda-nvrtc-cu12==12.8.93",
        "nvidia-cuda-runtime-cu12==12.8.90",
        "nvidia-cudnn-cu12==9.10.2.21",
        "nvidia-cufft-cu12==11.3.3.83",
        "nvidia-cufile-cu12==1.13.1.3",
        "nvidia-curand-cu12==10.3.9.90",
        "nvidia-cusolver-cu12==11.7.3.90",
        "nvidia-cusparse-cu12==12.5.8.93",
        "nvidia-cusparselt-cu12==0.7.1",
        "nvidia-nccl-cu12==2.27.3",
        "nvidia-nvjitlink-cu12==12.8.93",
        "nvidia-nvtx-cu12==12.8.90",
        "packaging==25.0",
        "peft==0.17.1",
        "pillow==11.3.0",
        "protobuf==6.32.1",
        "psutil==7.1.0",
        "pydantic==2.12.0",
        "pydantic-core==2.41.1",
        "pyparsing==3.2.5",
        "python-dateutil==2.9.0.post0",
        "python-dotenv==1.1.1",
        "PyYAML==6.0.3",
        "regex==2025.9.18",
        "requests==2.32.5",
        "safetensors==0.6.2",
        "sentencepiece==0.2.1",
        "setuptools==78.1.1",
        "six==1.17.0",
        "sniffio==1.3.1",
        "sympy==1.14.0",
        "tokenizers==0.21.4",
        "torch==2.8.0",
        "torchaudio==2.8.0",
        "torchvision==0.23.0",
        "tqdm==4.67.1",
        "transformers==4.49.0",
        "triton==3.4.0",
        "typing-inspection==0.4.2",
        "typing-extensions==4.15.0",
        "urllib3==2.5.0",
        "wheel==0.45.1",
        "zipp==3.23.0",
    )
    .apt_install("git")
    .run_commands(
        "huggingface-cli login --token $HF_TOKEN",
        secrets=[modal.Secret.from_name("huggingface-secret")],
    )
    .pip_install(
        "scipy",
        "langgraph",
        "fastapi[standard]"
    )
    .run_commands('git clone https://github.com/YashBhamare123/diffusers.git && pip install -e diffusers')
    .run_commands('cd diffusers && git pull origin main', force_build= True)
    .add_local_dir(
        local_path= '.',
        remote_path= '/root/files',
        ignore=[
        ".*",        
        "**/.*",
        "*.png",
        "*.jpg",
        "*.webp",
        "*.jpeg",
        "__pycache__/*",
        "images/*"
    ],
    )
)
@app.function(
    image=image,
    gpu="H100",
    timeout=600,
    volumes={
        "/cache": volume,
        "/files": files_volume,
    },
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("groq-secret"),
    ],
)

def run_tryon(subject_url: str, garment_url : str, control_url : str, config : dict, local_file = False):
    import os
    import sys
    
    os.environ["HF_HOME"] = "/cache"
    os.environ["TRANSFORMERS_CACHE"] = "/cache"
    os.environ["HF_HUB_CACHE"] = "/cache"
    os.environ["HF_ENABLE_PARALLEL_SHARD_DOWNLOAD"] = "1"
    os.environ['TORCH_HOME'] = '/cache/torch'
    
    os.makedirs("/cache/compile", exist_ok=True)
    sys.path.insert(0, "/root/files")

    import torch
    from main import TryOnPipeline, GenerateConfig
    params = GenerateConfig(**config)
    pipe = TryOnPipeline(params)
    out = pipe(subject_url, garment_url, control_url, local_file)
    return [out]

@app.function(image = image)
@modal.asgi_app()
def fastapi_app():
    import io
    from fastapi import FastAPI, Response
    from pydantic import BaseModel
    from typing import Union, Literal
    import torch
    webapp = FastAPI()

    class GenerateConfig(BaseModel):
        model_config = {"arbitrary_types_allowed": True}
        num_steps : int
        num_steps_logo : int
        seed : int
        sampler : Union[Literal['euler'], Literal['dmpp_3_sde']]
        flux_guidance : float
        cache_conditioning : bool = True
        # CFG : float
        device : str = 'cuda'
        dtype : torch.dtype = torch.bfloat16
        redux_strength : float
        logo_redux_strength : float
        ACE_scale : float
        compile_repeated : bool = False
        teacache_coeff : float = 0.1
        image_res : int = 1024
        grow_padding : int = 20
    
    class CreatePredictions(BaseModel):
        subject_url : str
        garment_url : str
        control_url : str
        config : GenerateConfig

    @webapp.post('/tryon')
    def tryon(data : CreatePredictions):
        out = run_tryon.remote(data.subject_url, data.garment_url, data.control_url, data.config.model_dump())
        buf = io.BytesIO()
        out[0].save(buf, format="PNG")

        return Response(
            content=buf.getvalue(),
            media_type="image/png"
        )
    return webapp

@app.local_entrypoint()
def main():
    import torch
    config = {
        "num_steps": 25,
        "num_steps_logo": 10,
        "seed": 39,
        "sampler": "euler",
        "flux_guidance": 40,
        # "CFG": 1.0,
        "redux_strength": 0.2,
        "logo_redux_strength": 0.2,
        "ACE_scale": 1.0,
        "dtype": torch.bfloat16,
        "compile_repeated": True
    }

    subject_url = "/files/subject.jpg"
    garment_url = "/files/garment.webp"
    control_url = "/files/control.png"
    image_bytes_list = run_tryon.remote(subject_url, garment_url, control_url, config, local_file = True)
    
    for img in image_bytes_list:
        img.save(f'images/outputs/out.png')
            
    print("Images saved as out.png")