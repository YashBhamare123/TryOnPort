import gradio as gr
import modal
import io
import cloudinary
import cloudinary.uploader
from PIL import Image
from dotenv import load_dotenv
import os

load_dotenv()

cloudinary.config(
  cloud_name = os.getenv("CLOUD_NAME"),
  api_key = os.getenv("CLOUDINARY_API_KEY"),
  api_secret = os.getenv("CLOUDINARY_API_SECRET")
)

tryon_cls = modal.Cls.from_name("tryon-inference", "TryOnInference")

def process_tryon(subject_file, subject_url_input, garment_file, garment_url_input):
    def get_url_and_id(file_obj, url_str):
        if url_str and url_str.strip():
            return url_str.strip(), None
        elif file_obj is not None:
            response = cloudinary.uploader.upload(file_obj)
            return response["secure_url"], response["public_id"]
        return None, None

    sub_url, sub_id = get_url_and_id(subject_file, subject_url_input)
    garm_url, garm_id = get_url_and_id(garment_file, garment_url_input)

    if not sub_url or not garm_url:
        return None

    try:
        # CHANGED: Instantiate the class () and call the method
        output_data = tryon_cls().run_tryon.remote(sub_url, garm_url)
        
        result_images = []
        for item in output_data:
            if isinstance(item, Image.Image):
                result_images.append(item)
            elif isinstance(item, (bytes, bytearray)):
                result_images.append(Image.open(io.BytesIO(item)))
            
        if result_images:
            return [result_images[0]]
        return []

    finally:
        if sub_id:
            cloudinary.uploader.destroy(sub_id)
        if garm_id:
            cloudinary.uploader.destroy(garm_id)

custom_css = """
/* --- Output Gallery Styling (Right Side) --- */
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

/* --- Input Image Styling (Left Side - Smaller) --- */
.input-image-class {
    max-height: 300px !important; /* Limits height of the container */
}

.input-image-class img {
    object-fit: contain !important; 
    max-height: 280px !important; /* Limits height of the actual image */
}

/* --- General UI --- */
.or-divider {
    text-align: center;
    font-weight: bold;
    color: #888;
    margin: 5px 0;
}
"""

with gr.Blocks(title="Virtual Try-On") as demo:
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

    run_button.click(
        fn=process_tryon,
        inputs=[subject_image, subject_url, garment_image, garment_url],
        outputs=output_gallery
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", 
                server_port=7860, 
                share=True,
                css=custom_css)