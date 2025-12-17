import gradio as gr
import modal
import io
import cloudinary
import cloudinary.uploader
from PIL import Image
from dotenv import load_dotenv
load_dotenv()

cloudinary.config(
  cloud_name = os.getenv("CLOUD_NAME"),
  api_key = os.getenv("CLOUDINARY_API_KEY"),
  api_secret = os.getenv("CLOUDINARY_API_SECRET")
)

tryon_fn = modal.Function.from_name("tryon-inference", "run_tryon")

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
        output_bytes_list = tryon_fn.remote(sub_url, garm_url)
        
        result_images = []
        for img_bytes in output_bytes_list:
            result_images.append(Image.open(io.BytesIO(img_bytes)))
            
        if result_images:
            return [result_images[0]]
        return []

    finally:
        if sub_id:
            cloudinary.uploader.destroy(sub_id)
        if garm_id:
            cloudinary.uploader.destroy(garm_id)

custom_css = """
.fixed-height-img img {
    max-height: 300px !important; 
    object-fit: contain !important;
}

.output-gallery-class {
    min-height: 500px !important;
}

.output-gallery-class img {
    object-fit: contain !important;
    max-height: 80vh !important;
    width: 100% !important;
}
"""

with gr.Blocks(css=custom_css) as demo:
    with gr.Row():
        with gr.Column():
            with gr.Tabs():
                with gr.TabItem("Upload Image"):
                    subject_image = gr.Image(label="Subject Image", type="filepath", elem_classes="fixed-height-img")
                with gr.TabItem("Paste URL"):
                    subject_url = gr.Textbox(label="Subject URL")

            with gr.Tabs():
                with gr.TabItem("Upload Image"):
                    garment_image = gr.Image(label="Garment Image", type="filepath", elem_classes="fixed-height-img")
                with gr.TabItem("Paste URL"):
                    garment_url = gr.Textbox(label="Garment URL")

            run_button = gr.Button("Run Try-On", variant="primary")
            
        with gr.Column():
            output_gallery = gr.Gallery(
                label="Result", 
                columns=1, 
                height="auto", 
                format="png",
                elem_classes="output-gallery-class",
                preview=True
            )

    run_button.click(
        fn=process_tryon,
        inputs=[subject_image, subject_url, garment_image, garment_url],
        outputs=output_gallery
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)