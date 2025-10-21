import torch
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from typing import List, Tuple
from torchvision.transforms import ToPILImage

class ProperLogo:
    def __init__(self):
        print("Initializing the detector...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.model.to(self.device)
        print(f"Model loaded successfully on device: {self.device}")

    def process_image(
        self,
        image: torch.Tensor,
        prompt: str,
        threshold: float = 0.2,
        detection_padding: int = 10,
        crop_padding: int = 10
    ) -> Tuple[List[Image.Image], List[Image.Image], List[Image.Image], List]:
        if image.shape[0] != 1:
            raise ValueError(f"Input batch size must be 1, but got {image.shape[0]}")
        pil_image = ToPILImage()(image)
        texts = [[p.strip() for p in prompt.split(',')]]
        width, height = pil_image.size

        inputs = self.processor(text=texts, images=pil_image, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.Tensor([pil_image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=threshold)

        boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]

        mask_pil = Image.new("L", pil_image.size, 0)
        draw_mask = ImageDraw.Draw(mask_pil)

        for box, score, label in zip(boxes, scores, labels):
            label_idx = int(label.item())
            label_text = texts[0][label_idx] if label_idx < len(texts[0]) else "unknown"
            box_orig = [round(i, 2) for i in box.tolist()]
            print(f"Detected {label_text} with confidence {round(score.item(), 3)} at {box_orig}")

            x1, y1, x2, y2 = box_orig
            x1_padded = max(0, x1 - detection_padding)
            y1_padded = max(0, y1 - detection_padding)
            x2_padded = min(width, x2 + detection_padding)
            y2_padded = min(height, y2 + detection_padding)

            draw_mask.rectangle([x1_padded, y1_padded, x2_padded, y2_padded], fill=255)

        mask_tensor = torch.from_numpy(np.array(mask_pil).astype(np.float32) / 255.0).unsqueeze(0)
        image_np = image.permute(1, 2, 0).cpu().numpy()
        mask_np = mask_tensor[0].cpu().numpy()
        mask_binary = (mask_np > 0.5).astype(np.uint8)

        structure = ndimage.generate_binary_structure(2, 2)
        labeled_array, num_features = ndimage.label(mask_binary, structure=structure)

        if num_features == 0:
            print("No objects detected.")
            empty_images = []
            empty_masks = []
            empty_boxes = []
            return empty_images, empty_masks, empty_masks, empty_boxes

        slices = ndimage.find_objects(labeled_array)
        cropped_images, cropped_masks, bounding_boxes = [], [], []

        H, W, _ = image_np.shape
        for slc in slices:
            y_slice, x_slice = slc
            y_start = max(0, y_slice.start - crop_padding)
            y_end = min(H, y_slice.stop + crop_padding)
            x_start = max(0, x_slice.start - crop_padding)
            x_end = min(W, x_slice.stop + crop_padding)

            cropped_images.append(image_np[y_start:y_end, x_start:x_end, :])
            cropped_masks.append(mask_np[y_start:y_end, x_start:x_end])
            bounding_boxes.append([x_start, y_start, x_end, y_end])

        target_height = 512
        resized_images, resized_masks = [], []
        for img, msk in zip(cropped_images, cropped_masks):
            h, w = img.shape[:2]
            if h == 0 or w == 0:
                continue
            aspect_ratio = w / h
            target_width = int(target_height * aspect_ratio)
            zoom_y = target_height / h
            zoom_x = target_width / w
            resized_images.append(ndimage.zoom(img, (zoom_y, zoom_x, 1), order=1))
            resized_masks.append(ndimage.zoom(msk, (zoom_y, zoom_x), order=0))

        final_target_width = min(img.shape[1] for img in resized_images)
        batched_images, batched_masks = [], []
        for img, msk in zip(resized_images, resized_masks):
            current_height, current_width = img.shape[:2]
            zoom_y = 1.0
            zoom_x = final_target_width / current_width
            final_img = ndimage.zoom(img, (zoom_y, zoom_x, 1), order=1)
            final_msk = ndimage.zoom(msk, (zoom_y, zoom_x), order=0)
            batched_images.append(final_img)
            batched_masks.append(final_msk)

        final_image_tensor = torch.from_numpy(np.stack(batched_images)).float().to(self.device)
        final_mask_tensor = torch.from_numpy(np.stack(batched_masks)).float().to(self.device)
        blank_mask_tensor = torch.zeros_like(final_mask_tensor)

        # Return lists of PIL images
        return (
            [ToPILImage()(img.cpu().permute(2, 0, 1)) for img in final_image_tensor],
            [ToPILImage()(msk.cpu().unsqueeze(0)) for msk in final_mask_tensor],
            [ToPILImage()(bm.cpu().unsqueeze(0)) for bm in blank_mask_tensor],
            bounding_boxes
        )
