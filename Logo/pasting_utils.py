import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from typing import List, Tuple

def pil_to_tensor(pil_image: Image.Image) -> torch.Tensor:
    """
    Converts a PIL Image to a PyTorch tensor in (C, H, W) format.
    """
    image_np = np.array(pil_image.convert("RGB"))
    # Transpose from (H, W, C) to (C, H, W)
    return torch.from_numpy(image_np).permute(2, 0, 1)

class ImagePaster:
    """
    A standalone class to sequentially paste a batch of crop images onto a
    single base image (provided as a tensor) and return the result as a tensor.
    """
    def __init__(self):
        pass

    def paste_images_sequentially(self,
                                  base_image_tensor: torch.Tensor,
                                  crop_images: List[Image.Image],
                                  bounding_boxes: List[Tuple[int, int, int, int]],
                                  blend_amount: float = 0.25,
                                  sharpen_amount: int = 0) -> torch.Tensor:

        if base_image_tensor.ndim != 3 or base_image_tensor.shape[0] not in [1, 3, 4]:
            raise ValueError("Input base_image_tensor must be 3-dimensional in (C, H, W) format.")

        if base_image_tensor.is_floating_point():
            base_image_tensor = base_image_tensor.mul(255).byte()

        image_np = base_image_tensor.cpu().permute(1, 2, 0).numpy()
        base_image = Image.fromarray(image_np)

        if not crop_images:
            return base_image_tensor

        composite_image = base_image.convert("RGBA")

        # 2. Perform all pasting operations in PIL domain (inside the loop)
        for i in range(len(crop_images)):
            crop_image = crop_images[i]
            box = bounding_boxes[i]
            left, top, right, bottom = box
            composite_image = self._paste_single_image(
                composite_image, crop_image, left, top, right, bottom,
                blend_amount, sharpen_amount
            )

        return pil_to_tensor(composite_image)

    def _paste_single_image(self, base_image: Image.Image, crop_image: Image.Image, left, top, right, bottom, blend_amount, sharpen_amount) -> Image.Image:
        """ 
        This internal helper function should always return a PIL Image
        to be used in the next iteration of the loop.
        """
        crop_image = crop_image.convert("RGBA")
        crop_size = (right - left, bottom - top)

        if crop_size[0] <= 0 or crop_size[1] <= 0:
            return base_image
            
        crop_img_resized = crop_image.resize(crop_size, Image.Resampling.LANCZOS)

        if sharpen_amount > 0:
            for _ in range(sharpen_amount):
                crop_img_resized = crop_img_resized.filter(ImageFilter.SHARPEN)

        pasted_layer = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
        pasted_layer.paste(crop_img_resized, (left, top))

        blend_ratio = (max(crop_size) / 2) * float(blend_amount)
        mask = Image.new("L", base_image.size, 0)
        mask_block = Image.new("L", crop_size, 255)
        
        if blend_ratio > 0:
            border_width = int(blend_ratio / 2)
            if border_width > 0 and crop_size[0] > 2 * border_width and crop_size[1] > 2 * border_width:
                draw = ImageDraw.Draw(mask_block)
                draw.rectangle(
                    (border_width, border_width, crop_size[0] - border_width, crop_size[1] - border_width),
                    fill=255, outline=0, width=0
                )
        mask.paste(mask_block, (left, top))
        
        if blend_ratio > 0:
            blur_radius = blend_ratio / 4
            if blur_radius > 0:
                mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        return Image.composite(pasted_layer, base_image, mask)