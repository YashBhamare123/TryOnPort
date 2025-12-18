
import modal
from pathlib import Path

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
        "pip install git+https://github.com/YashBhamare123/diffusers.git && huggingface-cli login --token $HF_TOKEN",
        secrets=[modal.Secret.from_name("huggingface-secret")],
    )
    .pip_install(
        "scipy",
        "langgraph"
    )
    .add_local_dir(
        local_path= '.',
        remote_path= '/root/files',
        ignore=[
        ".*",        
        "**/.*",     
    ],
    )
)

@app.cls(
    image=image,
    gpu="A100",
    timeout=600,
    volumes={"/cache": volume, "/files": files_volume},
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("groq-secret"),
    ],
    max_containers=1 
)
class TryOnInference:

    @modal.enter()
    def initialize_model(self):
        import os
        import sys
        
        # Set Env Variables
        os.environ["HF_HOME"] = "/cache"
        os.environ["TRANSFORMERS_CACHE"] = "/cache"
        os.environ["HF_HUB_CACHE"] = "/cache"
        os.environ["HF_ENABLE_PARALLEL_SHARD_DOWNLOAD"] = "1"
        os.environ['TORCH_HOME'] = '/cache/torch'
        
        # Setup paths
        os.makedirs("/cache/compile", exist_ok=True)
        sys.path.insert(0, "/root/files")

        import torch
        from sampling.config import GenerateConfig
        from main import TryOnPipeline
        
        # Load Config
        self.params = GenerateConfig(
            num_steps=25,
            num_steps_logo= 10,
            seed=42,
            sampler='euler',
            flux_guidance=40,
            CFG=1.0,
            redux_strength=0.7,
            logo_redux_strength=0.7,
            ACE_scale=1.0,
            dtype=torch.bfloat16,
            compile_repeated=True
        )

        self.pipeline = TryOnPipeline(self.params)
        print("Model initialized and cached in memory.")

    @modal.method()
    def run_tryon(self, subject_url: str, garment_url: str):
        out = self.pipeline(subject_url, garment_url)

        image_list = []
        for img in out:
            import io
            byte_arr = io.BytesIO()
            img.save(byte_arr, format='PNG')
            image_list.append(byte_arr.getvalue())
            
        return image_list

# @app.local_entrypoint()
# def main():
#     inference_service = TryOnInference() 
#     image_bytes_list = inference_service.run_tryon.remote(
#         "https://res.cloudinary.com/dukgi26uv/image/upload/v1754049601/tryon-images/fx3mo7u3n0i42tcod9qv.jpg",
#         "https://res.cloudinary.com/dukgi26uv/image/upload/v1759842480/Manchester_United_Home_92.94_Shirt_kyajol.webp"
#     )
#     for idx, img in enumerate(image_bytes_list):
#         with open(f'output_{idx}.png', 'wb') as f:
#             f.write(img)
#     print("Images saved as output_*.png")