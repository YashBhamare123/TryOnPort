import torch
import numpy as np
from PIL import Image, ImageFilter
from typing import List, Tuple
from torchvision.transforms.v2 import ToPILImage, ToDtype
import os
import time


class ImagePaster:
    """
    A utility class to sequentially paste cropped images onto a base image.
    Handles resizing, blending, and sharpening, with options to save intermediate
    steps for debugging.
    """

    def __init__(self, debug_dir: str = "debug_outputs"):
        """
        Initializes the ImagePaster.

        Args:
            debug_dir (str): The directory to save debugging outputs. A timestamped
                             subdirectory will be created inside this directory for each run.
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.debug_dir = os.path.join(debug_dir, f"run_{timestamp}")
        os.makedirs(self.debug_dir, exist_ok=True)
        print(f"[DEBUG] Intermediate pasting images will be saved to: {self.debug_dir}")

    def paste_images_sequentially(
        self,
        base_image_tensor: torch.Tensor,
        crop_tensors: torch.Tensor,
        bounding_boxes: List[Tuple[int, int, int, int]],
        blend_amount: float = 0.25,  # blend_amount is not used in the current pasting logic
        sharpen_amount: int = 0,
    ) -> torch.Tensor:
        """
        Pastes a batch of crop tensors onto a single base image tensor sequentially.

        Args:
            base_image_tensor (torch.Tensor): The base image, shape (1, C, H, W).
            crop_tensors (torch.Tensor): A batch of images to paste, shape (N, C, H, W).
            bounding_boxes (List[Tuple[int, int, int, int]]): Bounding boxes for each crop.
            blend_amount (float): (Currently unused) The amount of blending to apply.
            sharpen_amount (int): Number of times to apply a sharpening filter.

        Returns:
            torch.Tensor: The final composed image as a tensor, shape (1, C, H, W).
        """
        if base_image_tensor.ndim != 4 or base_image_tensor.shape[0] != 1:
            raise ValueError(f"Expected base image tensor of shape (1, C, H, W), got {base_image_tensor.shape}")
        if crop_tensors.shape[0] != len(bounding_boxes):
            raise ValueError("The 'crop_tensors' batch and 'bounding_boxes' list must have the same length.")

        if crop_tensors.shape[0] == 0:
            return base_image_tensor

        # Convert base tensor to PIL for pasting operations
        base_image_pil = ToPILImage()(base_image_tensor[0].cpu().to(torch.float32)).convert("RGBA")
        base_path = os.path.join(self.debug_dir, "00_base_image.png")
        base_image_pil.save(base_path)

        composite_image_pil = base_image_pil.copy()

        for i in range(len(crop_tensors)):
            crop_tensor = crop_tensors[i]
            box = bounding_boxes[i]
            left, top, right, bottom = map(int, box)

            crop_image_pil = ToPILImage()(crop_tensor.cpu().to(torch.float32)).convert("RGBA")
            crop_path = os.path.join(self.debug_dir, f"01_crop_raw_{i}.png")
            crop_image_pil.save(crop_path)

            composite_image_pil = self._paste_single_image(
                composite_image_pil, crop_image_pil, top, left, right, bottom, sharpen_amount, step_idx=i
            )
            composite_step_path = os.path.join(self.debug_dir, f"05_composite_after_step_{i}.png")
            composite_image_pil.save(composite_step_path)

        final_image_pil_rgb = composite_image_pil.convert("RGB")

        # Convert final PIL image back to a bfloat16 tensor
        to_bfloat16 = ToDtype(torch.bfloat16, scale=True)
        final_image_tensor = to_bfloat16(torch.from_numpy(np.array(final_image_pil_rgb)).permute(2, 0, 1))

        final_path = os.path.join(self.debug_dir, "06_final_result.png")
        final_image_pil_rgb.save(final_path)

        return final_image_tensor.unsqueeze(0).to(base_image_tensor.device)

    def _paste_single_image(self, base_image, crop_image, top, left, right, bottom, sharpen_amount=0, step_idx=0):
        """
        Helper function to paste a single crop image onto a base image.
        Uses premultiplied alpha to avoid black fringes on transparent edges.
        """
        crop_size = (right - left, bottom - top)
        if crop_size[0] <= 0 or crop_size[1] <= 0:
            return base_image

        # Premultiply alpha for high-quality resizing of transparent images
        crop_np = np.array(crop_image).astype(np.float32)
        alpha = crop_np[..., 3:4] / 255.0
        crop_np[..., :3] *= alpha
        crop_premult = Image.fromarray(np.uint8(crop_np.clip(0, 255)), mode="RGBA")

        crop_resized = crop_premult.resize(crop_size, Image.LANCZOS)

        # Un-premultiply alpha after resizing (FIXED)
        crop_np_resized = np.array(crop_resized).astype(np.float32)
        alpha_resized = crop_np_resized[..., 3:4]
        mask_nonzero = alpha_resized > 1e-6  # avoid division by zero

        # Safe broadcasting-based division
        denominator = np.where(mask_nonzero, alpha_resized / 255.0, 1.0)
        crop_np_resized[..., :3] /= denominator
        np.clip(crop_np_resized, 0, 255, out=crop_np_resized)

        crop_unpremult = Image.fromarray(np.uint8(crop_np_resized), mode="RGBA")

        if sharpen_amount > 0:
            for _ in range(sharpen_amount):
                crop_unpremult = crop_unpremult.filter(ImageFilter.SHARPEN)

        # Paste the crop using its alpha channel
        base_image.paste(crop_unpremult, (left, top), crop_unpremult)

        return base_image
