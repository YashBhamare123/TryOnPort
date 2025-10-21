import torch
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from typing import List, Tuple
from torchvision.transforms.v2 import ToPILImage, ToDtype, Compose, Resize
import torch.nn.functional as F


class ProperLogo:
    """
    A class to detect objects (logos) in an image using the OWLv2 model.
    """

    def __init__(self):
        """
        Initializes the detector by loading the OWLv2 model and processor.
        The model is set to evaluation mode and moved to the available device (GPU or CPU).
        """
        print("Initializing the detector...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16
        self.processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.model = model.to(self.device, dtype=self.dtype).eval()
        print(f"Model loaded successfully on device: {self.device}")

    def process_image(
        self,
        image: torch.Tensor,
        prompt: str,
        threshold: float = 0.2,
        detection_padding: int = 10,
        crop_padding: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[int]]]:
        """
        Processes a single image to detect objects based on a text prompt.

        The function takes a (1, C, H, W) image tensor, detects objects, and returns the cropped detections
        as tensor batches.

        Args:
            image (torch.Tensor): The input image tensor of shape (1, C, H, W) and dtype bfloat16.
            prompt (str): A comma-separated string of object names to detect.
            threshold (float): Confidence threshold for object detection.
            detection_padding (int): Padding to add to the bounding box for mask generation.
            crop_padding (int): Padding to add when cropping the detected objects.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[List[int]]]: A tuple containing:
                - final_image_tensor (torch.Tensor): Batch of cropped images of detected objects, shape (N, C, H, W).
                - final_mask_tensor (torch.Tensor): Batch of corresponding masks, shape (N, 1, H, W).
                - blank_mask_tensor (torch.Tensor): Batch of blank masks, same shape as final_mask_tensor.
                - bounding_boxes (List[List[int]]): List of bounding boxes for the detected objects.
        """
        if image.shape[0] != 1:
            raise ValueError(f"Input batch size must be 1, but got {image.shape[0]}")

        # Convert tensor to PIL Image for processing with the detection model
        pil_image = ToPILImage()(image[0].cpu().to(torch.float32))
        texts = [[p.strip() for p in prompt.split(',')]]
        width, height = pil_image.size

        inputs = self.processor(text=texts, images=pil_image, return_tensors='pt').to(self.device, self.dtype)
        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.Tensor([pil_image.size[::-1]]).to(self.device)
        results = self.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=threshold)

        boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]

        if boxes.numel() == 0:
            print("No objects detected.")
            empty_tensor = torch.empty(0, 3, 0, 0, device=self.device, dtype=self.dtype)
            empty_mask = torch.empty(0, 1, 0, 0, device=self.device, dtype=self.dtype)
            return empty_tensor, empty_mask, empty_mask, []

        mask_pil = Image.new("L", pil_image.size, 0)
        draw_mask = ImageDraw.Draw(mask_pil)
        for box, score, label in zip(boxes, scores, labels):
            box_orig = [round(i) for i in box.tolist()]
            x1, y1, x2, y2 = box_orig
            # Add padding to the detection mask
            draw_mask.rectangle([
                max(0, x1 - detection_padding), max(0, y1 - detection_padding),
                min(width, x2 + detection_padding), min(height, y2 + detection_padding)
            ], fill=255)

        mask_np = np.array(mask_pil, dtype=np.float32) / 255.0
        image_np = np.array(pil_image, dtype=np.float32) / 255.0

        # Label connected components in the binary mask
        labeled_array, num_features = ndimage.label(mask_np > 0.5, structure=ndimage.generate_binary_structure(2, 2))
        if num_features == 0:
            return torch.empty(0, 3, 0, 0), torch.empty(0, 1, 0, 0), torch.empty(0, 1, 0, 0), []

        slices = ndimage.find_objects(labeled_array)
        cropped_images_np, cropped_masks_np, bounding_boxes = [], [], []

        H, W = image_np.shape[:2]
        for slc in slices:
            y_slice, x_slice = slc
            # Add padding for cropping
            y_start, y_end = max(0, y_slice.start - crop_padding), min(H, y_slice.stop + crop_padding)
            x_start, x_end = max(0, x_slice.start - crop_padding), min(W, x_slice.stop + crop_padding)

            cropped_images_np.append(image_np[y_start:y_end, x_start:x_end, :])
            cropped_masks_np.append(mask_np[y_start:y_end, x_start:x_end])
            bounding_boxes.append([x_start, y_start, x_end, y_end])
        
        # Convert cropped numpy arrays to tensors and perform an initial resize
        to_bfloat16 = ToDtype(torch.bfloat16, scale=True)
        intermediate_images, intermediate_masks = [], []
        
        target_height = 512
        for img_np, mask_np in zip(cropped_images_np, cropped_masks_np):
            img_tensor = to_bfloat16(torch.from_numpy(img_np).permute(2, 0, 1))
            mask_tensor = to_bfloat16(torch.from_numpy(mask_np).unsqueeze(0))
            
            h, w = img_tensor.shape[1:]
            if h == 0 or w == 0: continue
            
            aspect_ratio = w / h
            target_width = int(target_height * aspect_ratio)

            resizer = Resize((target_height, target_width), antialias=True)
            intermediate_images.append(resizer(img_tensor))
            intermediate_masks.append(resizer(mask_tensor))

        if not intermediate_images:
             return torch.empty(0, 3, 0, 0), torch.empty(0, 1, 0, 0), torch.empty(0, 1, 0, 0), []

        # Find the smallest dimensions in the batch to use as the final uniform size
        min_height = min(t.shape[1] for t in intermediate_images)
        min_width = min(t.shape[2] for t in intermediate_images)
        
        final_resizer = Resize((min_height, min_width), antialias=True)
        
        final_images, final_masks = [], []
        for img, msk in zip(intermediate_images, intermediate_masks):
            final_images.append(final_resizer(img))
            final_masks.append(final_resizer(msk))

        batched_images = torch.stack(final_images)
        batched_masks = torch.stack(final_masks)

        final_image_tensor = batched_images.to(self.device, self.dtype)
        final_mask_tensor = batched_masks.to(self.device, self.dtype)
        blank_mask_tensor = torch.zeros_like(final_mask_tensor)
        print("final_image-tensor:",final_image_tensor.shape)
        print("final_mask_tensor:",final_mask_tensor.shape)
        print("blank_mask_tensor:",blank_mask_tensor.shape)
        print(bounding_boxes)
        return final_image_tensor, final_mask_tensor, blank_mask_tensor, bounding_boxes