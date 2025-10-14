import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from typing import List, Tuple

def pil_to_tensorss(pil_image: Image.Image) -> torch.Tensor:
    """
    Converts a PIL Image to a PyTorch tensor.
    Assumes input is a PIL Image and returns a (H, W, C) tensor of type uint8.
    """
    return torch.from_numpy(np.array(pil_image))

def concatenate_images_from_memory(images_a: List[Image.Image], images_b: List[Image.Image]) -> Tuple[torch.Tensor, List[int]]:
    """
    Concatenates two lists of images horizontally and batches them into a single tensor.

    This function takes two lists of PIL images, pads them to the same height,
    concatenates each pair, pads them all to the same width, and stacks them
    into a single batch tensor.

    Args:
        images_a: A list of PIL images.
        images_b: A second list of PIL images, of the same length as images_a.

    Returns:
        A tuple containing:
        - A single torch.Tensor representing the batch of concatenated images
          in (B, C, H, W) format.
        - A list of the original widths of the images from the first list.
    """
    if len(images_a) != len(images_b):
        raise ValueError(f"Input image lists must have the same number of images. "
                         f"Got {len(images_a)} for list A and {len(images_b)} for list B.")

    if not images_a:
        # Return an empty 4D tensor and an empty list if input is empty
        return torch.empty((0, 3, 0, 0)), []

    concatenated_tensors = []
    original_widths_a = []

    for img_a_pil, img_b_pil in zip(images_a, images_b):
        # Convert PIL images to (H, W, C) tensors
        img_a = pil_to_tensorss(img_a_pil.convert("RGB"))
        img_b = pil_to_tensorss(img_b_pil.convert("RGB"))

        original_widths_a.append(img_a.shape[1])
        
        h_a, w_a = img_a.shape[:2]
        h_b, w_b = img_b.shape[:2]
        target_height = max(h_a, h_b)

        # Permute to (C, H, W) for PyTorch operations
        img_a_chw = img_a.permute(2, 0, 1)
        img_b_chw = img_b.permute(2, 0, 1)

        # Calculate and apply vertical padding to match heights
        pad_a_total = target_height - h_a
        pad_a_top, pad_a_bottom = pad_a_total // 2, pad_a_total - (pad_a_total // 2)
        padded_a = F.pad(img_a_chw, (0, 0, pad_a_top, pad_a_bottom), "constant", 0)

        pad_b_total = target_height - h_b
        pad_b_top, pad_b_bottom = pad_b_total // 2, pad_b_total - (pad_b_total // 2)
        padded_b = F.pad(img_b_chw, (0, 0, pad_b_top, pad_b_bottom), "constant", 0)

        # Concatenate tensors horizontally along the width dimension
        concatenated_chw = torch.cat((padded_a, padded_b), dim=2)
        concatenated_tensors.append(concatenated_chw)

    # Pad all concatenated tensors to the same width to enable batching
    max_width = max(t.shape[2] for t in concatenated_tensors)
    
    padded_batch = []
    for tensor in concatenated_tensors:
        width_pad = max_width - tensor.shape[2]
        # Pad on the right side of the width dimension (pad_left, pad_right, ...)
        padded_tensor = F.pad(tensor, (0, width_pad, 0, 0), "constant", 0)
        padded_batch.append(padded_tensor)

    # Stack the list of tensors into a single batch tensor
    final_batch_tensor = torch.stack(padded_batch, dim=0)

    return final_batch_tensor, original_widths_a


def pil_mask_to_tensor(pil_image: Image.Image) -> torch.Tensor:
    """
    Converts a PIL Image mask to a PyTorch tensor.

    Converts the image to grayscale ('L'), adds a channel dimension,
    and returns a (H, W, 1) tensor of type uint8.
    """
    mask_np = np.array(pil_image.convert("L"))
    # Add a channel dimension to be explicit: (H, W) -> (H, W, 1)
    mask_np_expanded = np.expand_dims(mask_np, axis=2)
    return torch.from_numpy(mask_np_expanded)


def concatenate_masks_from_memory(masks_a: List[Image.Image], masks_b: List[Image.Image]) -> Tuple[torch.Tensor, List[int]]:
    """
    Concatenates two lists of masks horizontally and batches them into a single tensor.

    This function takes two lists of PIL image masks, pads them to the same height,
    concatenates each pair, pads them all to the same width, and stacks them
    into a single batch tensor.

    Args:
        masks_a: A list of PIL image masks.
        masks_b: A second list of PIL image masks, of the same length as masks_a.

    Returns:
        A tuple containing:
        - A single torch.Tensor representing the batch of concatenated masks
          in (B, 1, H, W) format.
        - A list of the original widths of the masks from the first list.
    """
    if len(masks_a) != len(masks_b):
        raise ValueError(f"Input mask lists must have the same number of items. "
                         f"Got {len(masks_a)} for list A and {len(masks_b)} for list B.")

    if not masks_a:
        # Return an empty 4D tensor and an empty list if input is empty
        return torch.empty((0, 1, 0, 0)), []

    concatenated_tensors = []
    original_widths_a = []

    for mask_a_pil, mask_b_pil in zip(masks_a, masks_b):
        # Convert PIL masks to (H, W, 1) tensors
        mask_a = pil_mask_to_tensor(mask_a_pil)
        mask_b = pil_mask_to_tensor(mask_b_pil)
        
        original_widths_a.append(mask_a.shape[1])
        
        h_a, w_a = mask_a.shape[:2]
        h_b, w_b = mask_b.shape[:2]
        target_height = max(h_a, h_b)
        
        # Permute to (C, H, W) for PyTorch operations
        mask_a_chw = mask_a.permute(2, 0, 1)
        mask_b_chw = mask_b.permute(2, 0, 1)

        # Calculate and apply vertical padding to match heights
        pad_a_total = target_height - h_a
        pad_a_top, pad_a_bottom = pad_a_total // 2, pad_a_total - (pad_a_total // 2)
        padded_a = F.pad(mask_a_chw, (0, 0, pad_a_top, pad_a_bottom), "constant", 0)

        pad_b_total = target_height - h_b
        pad_b_top, pad_b_bottom = pad_b_total // 2, pad_b_total - (pad_b_total // 2)
        padded_b = F.pad(mask_b_chw, (0, 0, pad_b_top, pad_b_bottom), "constant", 0)
        
        # Concatenate tensors horizontally along the width dimension
        concatenated_chw = torch.cat((padded_a, padded_b), dim=2)
        concatenated_tensors.append(concatenated_chw)

    # Pad all concatenated tensors to the same width to enable batching
    max_width = max(t.shape[2] for t in concatenated_tensors)
    
    padded_batch = []
    for tensor in concatenated_tensors:
        width_pad = max_width - tensor.shape[2]
        # Pad on the right side of the width dimension
        padded_tensor = F.pad(tensor, (0, width_pad, 0, 0), "constant", 0)
        padded_batch.append(padded_tensor)

    # Stack the list of tensors into a single batch tensor
    final_batch_tensor = torch.stack(padded_batch, dim=0)

    return final_batch_tensor, original_widths_a

def deconcatenate_batch_horizontally(concatenated_images: List[Image.Image], original_widths_a: List[int]) -> Tuple[List[Image.Image], List[Image.Image]]:
    """
    Splits a list of horizontally concatenated images based on a list of original widths.
    """
    if len(concatenated_images) != len(original_widths_a):
        raise ValueError("The 'concatenated_images' list and 'original_widths_a' list must have the same number of items.")

    images_a = []
    images_b = []
    
    for i in range(len(concatenated_images)):
        img = concatenated_images[i]
        width_a = original_widths_a[i]
        
        # --- Integrated logic from the single-image function ---
        total_width, total_height = img.size
        if width_a > total_width:
            raise ValueError(f"Item {i}: The 'original_width_a' ({width_a}) cannot be greater than the total image width ({total_width}).")
        
        part_a = img.crop((0, 0, width_a, total_height))
        part_b = img.crop((width_a, 0, total_width, total_height))
        # --- End of integrated logic ---

        images_a.append(part_a)
        images_b.append(part_b)
        
    return images_a, images_b
