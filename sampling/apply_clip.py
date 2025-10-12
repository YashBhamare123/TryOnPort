import torch
from transformers import AutoProcessor, SiglipVisionModel
from PIL import Image
import requests

device = "cuda" if torch.cuda.is_available() else "cpu"

def clip_process(clip_path):

    try:   
        processor = AutoProcessor.from_pretrained(clip_path)
        model = SiglipVisionModel.from_pretrained(clip_path).to(device)
        model.eval()
        print("Model and processor loaded successfully.")
        return processor, model
    except OSError:
        print(f"Error: Could not find a valid model directory at '{clip_path}'.")
        print("Please ensure you have run the download script first and the directory exists.")
        exit()

def run_clip(image, clip_path):
    try:
        processor, model = clip_process(clip_path)

        print("\nEncoding a sample image...")

        with torch.no_grad():
            inputs = processor(images=image, return_tensors="pt").to(device)
            outputs = model(**inputs)
            clip_vision_output = outputs.last_hidden_state
            # image_embeds = outputs.pooler_output

        print("--- Success! ---")
        print(f"Shape of the final image embedding tensor: {clip_vision_output.shape}")
        return clip_vision_output
    except Exception as e:
        print(f"\nAn error occurred during inference: {e}")

if __name__ == "__main__":
    out = run_clip()