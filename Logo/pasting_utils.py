import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from typing import List, Tuple
from torchvision.transforms import ToPILImage, ToTensor
import os
import time


class ImagePaster:
    """
    Sequentially paste crop images onto a base image tensor (C, H, W)
    and return the result as a color tensor (C, H, W).
    Saves every intermediate step for debugging.
    """

    def __init__(self, debug_dir: str = "debug_outputs"):
        # Create a debug folder with a unique timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.debug_dir = os.path.join(debug_dir, f"run_{timestamp}")
        os.makedirs(self.debug_dir, exist_ok=True)
        print(
            f"[DEBUG] All intermediate images will be saved to: {self.debug_dir}")

    def paste_images_sequentially(
        self,
        base_image_tensor: torch.Tensor,
        crop_images: List[Image.Image],
        bounding_boxes: List[Tuple[int, int, int, int]],
        blend_amount: float = 0.25,
        sharpen_amount: int = 0
    ) -> torch.Tensor:

        print("base_image_tensor:", base_image_tensor.shape)

        if base_image_tensor.ndim != 3:
            raise ValueError(
                f"Expected base image tensor of shape (C, H, W), got {base_image_tensor.shape}")

        if not crop_images:
            return base_image_tensor

        crop_tensors = torch.stack([ToTensor()(img) for img in crop_images])
        print("crop_images_pasting:", crop_tensors.shape)

        if crop_tensors.shape[0] != len(bounding_boxes):
            raise ValueError(
                "The 'crop_images' and 'bounding_boxes' lists must have the same length.")

        # Save base image for debugging
        base_image_pil = ToPILImage()(base_image_tensor.clamp(0, 1)).convert("RGBA")
        base_path = os.path.join(self.debug_dir, "00_base_image.png")
        base_image_pil.save(base_path)
        print(f"[DEBUG] Saved base image → {base_path}")

        composite_image_pil = base_image_pil.copy()
        bboxes = torch.tensor(bounding_boxes, dtype=torch.int,
                              device=base_image_tensor.device)

        for i in range(len(crop_tensors)):
            crop_image_tensor = crop_tensors[i]
            box = bboxes[i]
            left, top, right, bottom = map(int, box.tolist())

            crop_image_pil = ToPILImage()(crop_image_tensor.clamp(0, 1)).convert("RGBA")

            # Save raw crop before resizing
            crop_path = os.path.join(self.debug_dir, f"01_crop_raw_{i}.png")
            crop_image_pil.save(crop_path)
            print(f"[DEBUG] Saved raw crop {i} → {crop_path}")

            composite_image_pil = self.paste_single_image(
                composite_image_pil,
                crop_image_pil,
                top, left, right, bottom,
                blend_amount,
                sharpen_amount,
                step_idx=i
            )

            # Save composite after each step
            composite_step_path = os.path.join(
                self.debug_dir, f"05_composite_after_step_{i}.png")
            composite_image_pil.save(composite_step_path)
            print(
                f"[DEBUG] Saved composite image after step {i} → {composite_step_path}")

        final_image_pil_rgb = composite_image_pil.convert("RGB")
        final_image_tensor = ToTensor()(final_image_pil_rgb).to(base_image_tensor.device)

        # Save final result
        final_path = os.path.join(self.debug_dir, "06_final_result.png")
        final_image_pil_rgb.save(final_path)
        print(f"[DEBUG] Saved final result → {final_path}")

        return final_image_tensor

    def paste_single_image(self, base_image, crop_image, top, left, right, bottom, sharpen_amount=0, step_idx=0):
        """Pastes one crop onto the base without introducing black edge lines."""
        crop_image = crop_image.convert("RGBA")
        crop_size = (right - left, bottom - top)
        if crop_size[0] <= 0 or crop_size[1] <= 0:
            return base_image

        # --- Premultiply alpha ---
        crop_np = np.array(crop_image).astype(np.float32)
        alpha = crop_np[..., 3:4] / 255.0
        crop_np[..., :3] *= alpha
        crop_premult = Image.fromarray(np.uint8(crop_np.clip(0, 255)), mode="RGBA")

        # Resize
        crop_resized = crop_premult.resize(crop_size, Image.LANCZOS)

        # Unpremultiply alpha
        crop_np_resized = np.array(crop_resized).astype(np.float32)
        alpha_resized = crop_np_resized[..., 3:4]
        mask_nonzero = alpha_resized > 0
        crop_np_resized[...,:3][mask_nonzero] /= (alpha_resized[mask_nonzero] / 255.0)
        crop_np_resized[..., :3] = np.clip(crop_np_resized[..., :3], 0, 255)
        crop_resized = Image.fromarray(np.uint8(crop_np_resized), mode="RGBA")
    
        # Optional sharpening
        for _ in range(max(0, sharpen_amount)):
            crop_resized = crop_resized.filter(ImageFilter.SHARPEN)

    # --- Paste using crop alpha as mask directly ---
        base_image.paste(crop_resized, (left, top), crop_resized)

        return base_image
