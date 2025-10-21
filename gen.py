import modal
from pathlib import Path
import os
import hashlib
import json

app = modal.App("tryon-inference")

volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

# Add a volume for tracking file hashes
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
        "scipy"
    )
)

@app.function(
    image=image,
    gpu="A100",
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
def run_tryon(files: dict, file_hashes: dict):
    import os
    import sys
    import importlib.util
    import json
    
    os.environ["HF_HOME"] = "/cache"
    os.environ["TRANSFORMERS_CACHE"] = "/cache"
    os.environ["HF_HUB_CACHE"] = "/cache"
    os.environ["HF_ENABLE_PARALLEL_SHARD_DOWNLOAD"] = "1"
    os.environ['TORCH_HOME'] = '/cache/torch'
    
    os.makedirs("/cache/compile", exist_ok=True)
    
    # Load previous hashes
    hash_file = "/files/file_hashes.json"
    previous_hashes = {}
    if os.path.exists(hash_file):
        with open(hash_file, "r") as f:
            previous_hashes = json.load(f)
    
    # Write only changed files
    files_written = 0
    files_skipped = 0
    for filename, content in files.items():
        current_hash = file_hashes[filename]
        
        # Check if file needs updating
        if filename in previous_hashes and previous_hashes[filename] == current_hash:
            if os.path.exists(f"/files/{filename}"):
                files_skipped += 1
                continue
        
        print(f"Writing {filename}...")
        files_written += 1
        
        # Write to /files volume
        directory = os.path.dirname(f"/files/{filename}")
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        with open(f"/files/{filename}", "w") as f:
            f.write(content)
    
    print(f"Files written: {files_written}, skipped: {files_skipped}")
    
    # Save current hashes
    with open(hash_file, "w") as f:
        json.dump(file_hashes, f)
    
    files_volume.commit()
    
    # Change to /files directory to run the code
    os.chdir("/files")
    
    # Add /files to Python path so imports work
    sys.path.insert(0, "/files")
    
    spec = importlib.util.spec_from_file_location("main", "/files/main.py")
    main_module = importlib.util.module_from_spec(spec)
    sys.modules["main"] = main_module
    spec.loader.exec_module(main_module)
    
    print("Running generate() from main.py...")
    
    import torch
    from main import generate, GenerateConfig
    
    params = GenerateConfig(
        num_steps=25,
        seed=42,
        sampler='euler',
        scheduler='simple',
        flux_guidance=30,
        CFG=1.,
        redux_strength=0.4,
        redux_strength_type="multiply",
        ACE_scale=1.,
        dtype=torch.bfloat16,
        compile_repeated= True
    )

    subject_url = "https://res.cloudinary.com/dukgi26uv/image/upload/v1759842454/the-nude-v-neck-pointelle-knit-tee-tops-snhkxv_2048x_bfnch4.webp"
    garment_url = "https://res.cloudinary.com/dukgi26uv/image/upload/v1759842480/Manchester_United_Home_92.94_Shirt_kyajol.webp"

    generate(subject_url, garment_url, params)
    
    image_list = []
    with open("output_fill_1.png", "rb") as f:
        image_list.append(f.read())
    with open("mask_image.png", "rb") as f:
        image_list.append(f.read())
    with open("subject_image.png", "rb") as f:
        image_list.append(f.read())
    
    return image_list

def compute_file_hash(filepath):
    """Compute MD5 hash of a file."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()

@app.local_entrypoint()
def main():
    import glob
    
    print("Starting TryOn on Modal...")
    
    files = {}
    file_hashes = {}
    
    # Include Python files
    for py_file in glob.glob("**/*.py", recursive=True):
        if py_file == "modal_inference.py" or py_file == os.path.basename(__file__):
            continue
        
        if "__pycache__" in py_file or "/." in py_file or py_file.startswith("."):
            continue
            
        with open(py_file, "r", encoding="utf-8") as f:
            content = f.read()
            files[py_file] = content
            file_hashes[py_file] = hashlib.md5(content.encode()).hexdigest()
        print(f"Including {py_file}")
    
    # Include other files (txt, json, etc.)
    for pattern in ["**/*.txt", "**/*.json", "**/*.yaml", "**/*.yml"]:
        for file_path in glob.glob(pattern, recursive=True):
            if "__pycache__" in file_path or "/." in file_path or file_path.startswith("."):
                continue
            
            try:
                with open(file_path, "r") as f:
                    content = f.read()
                    files[file_path] = content
                    file_hashes[file_path] = hashlib.md5(content.encode()).hexdigest()
                print(f"Including {file_path}")
            except Exception as e:
                print(f"Skipping {file_path}: {e}")
    
    if not files:
        print("Warning: No files found!")
    else:
        print(f"Total files tracked: {len(files)}")
    
    image_bytes_list = run_tryon.remote(files, file_hashes)
    
    for idx, img_bytes in enumerate(image_bytes_list):
        with open(f"output_{idx}.png", "wb") as f:
            f.write(img_bytes)
    
    print("Images saved as output_*.png")