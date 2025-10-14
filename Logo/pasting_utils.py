import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from typing import List, Tuple
from torchvision.transforms import ToPILImage, ToTensor

def tensor2pil(image: torch.Tensor) -> Image.Image:
    """Converts a torch tensor to a PIL image."""
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image: Image.Image) -> torch.Tensor:
    """Converts a PIL image to a torch tensor."""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


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

        if base_image_tensor.shape[0] != 1:
            raise ValueError("The base 'image' input must have a batch size of 1.")
        crop_imagess = torch.stack([ToTensor()(img) for img in crop_images])
        if crop_imagess.shape[0] != len(bounding_boxes):
            raise ValueError("The 'crop_images' batch and 'bounding_boxes' batch must have the same size.")

        batch_size = crop_imagess.shape[0]
        if batch_size == 0:
            return (base_image_tensor)

        # Start with the original image in PIL format, converted to RGBA for blending
        composite_image_pil = tensor2pil(base_image_tensor).convert("RGBA")
        for i in range(batch_size):
            crop_image_tensor=crop_imagess[i:i+1]
            box = bounding_boxes[i]
            left, top, right, bottom = box
            
            crop_image_pil=tensor2pil(crop_image_tensor)
            
            composite_image_pil=self._paste_single_image(
                composite_image_pil,
                crop_image_pil,
                top, left, right, bottom,
                blend_amount,
                sharpen_amount
            )
            
        final_image_pil_rgb=composite_image_pil.convert("RGB")
        final_image_tensor=pil2tensor(final_image_pil_rgb).to(base_image_tensor.device)
        return final_image_tensor

    def _paste_single_image(self, base_image, crop_image, top, left, right, bottom, blend_amount, sharpen_amount):
        """ Pastes a single crop onto the base image using blending. Assumes RGBA images. """
        crop_image = crop_image.convert("RGBA")

        crop_size = (right - left, bottom - top)
        if crop_size[0] <= 0 or crop_size[1] <= 0:
            return base_image
            
        crop_img_resized = crop_image.resize(crop_size, Image.LANCZOS)

        if sharpen_amount > 0:
            for _ in range(sharpen_amount):
                crop_img_resized = crop_img_resized.filter(ImageFilter.SHARPEN)

        # Create a new transparent layer for the crop to be pasted on
        pasted_layer = Image.new("RGBA", base_image.size, (0,0,0,0))
        pasted_layer.paste(crop_img_resized, (left, top))

        # Create the feathered mask for blending
        blend_ratio = (max(crop_size) / 2) * float(blend_amount)
        
        mask = Image.new("L", base_image.size, 0)
        mask_block = Image.new("L", crop_size, 255)
        
        if blend_ratio > 0:
            border_width = int(blend_ratio / 2)
            if border_width > 0:
                draw = ImageDraw.Draw(mask_block)
                draw.rectangle((0, 0, crop_size[0]-1, crop_size[1]-1), outline=0, width=border_width)
        
        mask.paste(mask_block, (left, top))
        
        if blend_ratio > 0:
            blur_radius = blend_ratio / 4
            if blur_radius > 0:
                mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            
        # Composite the base image with the pasted layer using the feathered mask
        return Image.composite(pasted_layer, base_image, mask)