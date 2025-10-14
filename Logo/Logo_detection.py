import torch
import numpy as np
from PIL import Image
from scipy import ndimage
from transformers import Owlv2Processor, Owlv2ForObjectDetection


class LogoBatchCropperTensor:
    def __init__(self):
        print("Initializing the detector...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.model.to(self.device)
        print(f"Model loaded successfully on device: {self.device}")

    def process_image(self, image_tensor: torch.Tensor, prompt: str, threshold: float, detection_padding: int, crop_padding: int):

        if image_tensor.ndim != 3 or image_tensor.shape[0] not in [1, 3, 4]:
            raise ValueError("Input tensor must be 3-dimensional in (C, H, W) format.")

        print("\nProcessing image tensor...")

        if image_tensor.is_floating_point():
            image_tensor = image_tensor.mul(255).byte()

        image_np_initial = image_tensor.cpu().permute(1, 2, 0).numpy()
        pil_image = Image.fromarray(image_np_initial).convert("RGB")
        
        original_width, original_height = pil_image.size

        texts = [[p.strip() for p in prompt.split(',')]]
        inputs = self.processor(text=texts, images=pil_image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([pil_image.size[::-1]], device=self.device)
        results = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=threshold)
        boxes = results[0]["boxes"]

        if len(boxes) == 0:
            print("No objects detected above the threshold.")
            return [], [], [], []
            
        print(f"Detected {len(boxes)} potential objects.")

        mask_np = np.zeros((original_height, original_width), dtype=np.float32)
        for box in boxes:
            x1, y1, x2, y2 = [round(i) for i in box.tolist()]
            x1p = max(0, x1 - detection_padding)
            y1p = max(0, y1 - detection_padding)
            x2p = min(original_width, x2 + detection_padding)
            y2p = min(original_height, y2 + detection_padding)
            mask_np[y1p:y2p, x1p:x2p] = 1.0

        structure = ndimage.generate_binary_structure(2, 2)
        labeled_array, num_features = ndimage.label(mask_np, structure=structure)

        if num_features == 0:
            print("Could not find any distinct regions in the mask.")
            return [], [], [], []
            
        print(f"Found {num_features} distinct region(s) to crop.")
        slices = ndimage.find_objects(labeled_array)
        image_np = np.array(pil_image)
        
        cropped_images_np = []
        cropped_masks_np = []
        bounding_boxes = []
        for slc in slices:
            y_slice, x_slice = slc
            y_start = max(0, y_slice.start - crop_padding)
            y_end = min(original_height, y_slice.stop + crop_padding)
            x_start = max(0, x_slice.start - crop_padding)
            x_end = min(original_width, x_slice.stop + crop_padding)
            
            cropped_images_np.append(image_np[y_start:y_end, x_start:x_end])
            cropped_masks_np.append(mask_np[y_start:y_end, x_start:x_end])
            bounding_boxes.append([x_start, y_start, x_end, y_end])

        target_height = 512
        resized_images, resized_masks = [], []
        for img, msk in zip(cropped_images_np, cropped_masks_np):
            h, w = img.shape[:2]
            if h == 0 or w == 0: continue
            aspect_ratio = w / h
            target_width = int(target_height * aspect_ratio)
            resized_images.append(ndimage.zoom(img, (target_height/h, target_width/w, 1), order=1))
            resized_masks.append(ndimage.zoom(msk, (target_height/h, target_width/w), order=0))

        if not resized_images: return [], [], [], []

        final_target_width = min(img.shape[1] for img in resized_images)
        final_images_np, final_masks_np = [], []
        
        for img, msk in zip(resized_images, resized_masks):
            current_width = img.shape[1]
            final_images_np.append(ndimage.zoom(img, (1.0, final_target_width/current_width, 1), order=1))
            final_masks_np.append(ndimage.zoom(msk, (1.0, final_target_width/current_width), order=0))

        blank_masks_np = [np.zeros_like(m, dtype=np.uint8) for m in final_masks_np]
        pil_images = [Image.fromarray(img) for img in final_images_np]
        pil_masks = [Image.fromarray((msk * 255).astype(np.uint8)) for msk in final_masks_np]
        pil_blanks = [Image.fromarray(b) for b in blank_masks_np]

        return pil_images, pil_masks, pil_blanks, bounding_boxes