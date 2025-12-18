import gradio as gr
import requests 
import io
import cloudinary
import cloudinary.uploader
from PIL import Image
from dotenv import load_dotenv
import os
import asyncio
from fastapi import FastAPI

load_dotenv()

app = FastAPI()

cloudinary.config(
    cloud_name=os.getenv("CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)


API_URL = "https://me240003014--tryon-inference-fastapi-app.modal.run/generate" 

async def process_tryon(subject_file, subject_url_input, garment_file, garment_url_input):
    def get_url_and_id_sync(file_obj, url_str):
        if url_str and url_str.strip():
            return url_str.strip(), None
        elif file_obj is not None:
            response = cloudinary.uploader.upload(file_obj)
            return response["secure_url"], response["public_id"]
        return None, None

    sub_url, sub_id = await asyncio.to_thread(get_url_and_id_sync, subject_file, subject_url_input)
    garm_url, garm_id = await asyncio.to_thread(get_url_and_id_sync, garment_file, garment_url_input)

    if not sub_url or not garm_url:
        return None

    try:
        payload = {
            "subject_url": sub_url,
            "garment_url": garm_url
        }

        response = await asyncio.to_thread(requests.post, API_URL, json=payload, timeout=600)
        
        if response.status_code == 200:
            image_bytes = response.content
            return [Image.open(io.BytesIO(image_bytes))]
        else:
            print(f"API Error: {response.status_code} - {response.text}")
            return []
            
    except Exception as e:
        print(f"Connection Error: {e}")
        return []

    finally:
        if sub_id:
            await asyncio.to_thread(cloudinary.uploader.destroy, sub_id)
        if garm_id:
            await asyncio.to_thread(cloudinary.uploader.destroy, garm_id)

custom_css = """
.output-gallery-class {
    height: 75vh !important; 
    min-height: 500px !important;
    display: flex;
    flex-direction: column;
}

.output-gallery-class .grid-wrap, 
.output-gallery-class .grid-container {
    height: 100% !important;
}

.output-gallery-class img {
    height: 100% !important;
    width: 100% !important;
    object-fit: contain !important;
    object-position: center !important;
    display: block;
}

.output-gallery-class .thumbnails {
    display: none !important;
}

.input-image-class {
    max-height: 300px !important;
}

.input-image-class img {
    object-fit: contain !important; 
    max-height: 280px !important;
}

.or-divider {
    text-align: center;
    font-weight: bold;
    color: #888;
    margin: 5px 0;
}
"""

with gr.Blocks(title="Virtual Try-On", css=custom_css) as demo:
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### Subject Image")
                subject_image = gr.Image(label="Upload Subject", type="filepath", elem_classes="input-image-class")
                gr.HTML("<div class='or-divider'>— OR —</div>")
                subject_url = gr.Textbox(label="Paste Subject URL", placeholder="https://...")

            with gr.Group():
                gr.Markdown("### Garment Image")
                garment_image = gr.Image(label="Upload Garment", type="filepath", elem_classes="input-image-class")
                gr.HTML("<div class='or-divider'>— OR —</div>")
                garment_url = gr.Textbox(label="Paste Garment URL", placeholder="https://...")

        with gr.Column(scale=1):
            output_gallery = gr.Gallery(
                label="Result", 
                columns=1, 
                rows=1,
                show_label=True,
                elem_classes="output-gallery-class",
                preview=True, 
                interactive=False
            )
            run_button = gr.Button("Run Try-On", variant="primary", size="lg")

    demo.queue(default_concurrency_limit=None)
    run_button.click(
        fn=process_tryon,
        inputs=[subject_image, subject_url, garment_image, garment_url],
        outputs=output_gallery
    )

app = gr.mount_gradio_app(app, demo, path="/")