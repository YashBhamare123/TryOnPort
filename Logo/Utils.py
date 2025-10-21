import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from typing import List, Tuple
from torchvision.transforms.v2 import ToDtype

def pil_to_tensor(pil_image: Image.Image, device: str = 'cpu') -> torch.Tensor:
    """
    Converts a single PIL Image to a PyTorch tensor with the specified schema.

    Args:
        pil_image (Image.Image): The input PIL image.
        device (str): The torch device to store the tensor on ('cpu' or 'cuda').

    Returns:
        torch.Tensor: The converted image tensor of shape (1, C, H, W) and dtype bfloat16.
    """
    image_np = np.array(pil_image.convert("RGB"))
    # Convert HWC uint8 to BCHW bfloat16
    tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
    return ToDtype(torch.bfloat16, scale=True)(tensor).to(device)


def concatenate_tensors(tensors_a: torch.Tensor, tensors_b: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
    """
    Concatenates two batches of tensors horizontally.

    The function pads tensors to the same height, concatenates them along the width dimension,
    and then pads all concatenated tensors to a uniform width to create a single batch.

    Args:
        tensors_a (torch.Tensor): The first batch of tensors, shape (B, C, H, W).
        tensors_b (torch.Tensor): The second batch of tensors, shape (B, C, H, W).

    Returns:
        Tuple[torch.Tensor, List[int]]: A tuple containing:
            - The concatenated batch tensor of shape (B, C, H_max, W_new).
            - A list of the original widths of the tensors in the first batch.
    """
    print("tensor_a_conc:",tensors_a.shape)
    print("tensors_b_conc:",tensors_b.shape)
    print("tensors[3]:",tensors_a.shape[3])
    if tensors_a.shape[0] != tensors_b.shape[0]:
        raise ValueError("Input tensor batches must have the same number of items.")
    if tensors_a.shape[0] == 0:
        return torch.empty(0, 3, 0, 0, dtype=tensors_a.dtype, device=tensors_a.device), []

    concatenated_tensors = []
    original_widths_a =[]
    for i, img in enumerate(tensors_a):
        original_widths_a.append(img.shape[2])
    print("original_widths_A",original_widths_a)
    for tensor_a, tensor_b in zip(tensors_a, tensors_b):
        print("tensor_a.shape",tensor_a.shape)
        h_a, h_b = tensor_a.shape[1], tensor_b.shape[1]
        target_height = max(h_a, h_b)

        # Pad height
        pad_a_total = target_height - h_a
        pad_a_top, pad_a_bottom = pad_a_total // 2, pad_a_total - (pad_a_total // 2)
        pad_b_total = target_height - h_b
        pad_b_top, pad_b_bottom = pad_b_total // 2, pad_b_total - (pad_b_total // 2)
        padded_a = F.pad(tensor_a, (0, 0, pad_a_top, pad_a_bottom),"constant",0)
        padded_b = F.pad(tensor_b, (0, 0, pad_b_top, pad_b_bottom),"constant",0)

        concatenated_tensors.append(torch.cat((padded_a, padded_b), dim=2))

    max_width = max(t.shape[2] for t in concatenated_tensors)
    padded_batch = [F.pad(t, (0, max_width - t.shape[2], 0, 0)) for t in concatenated_tensors]

    return torch.stack(padded_batch), original_widths_a


def deconcatenate_tensors(concatenated_tensor: torch.Tensor, original_widths_a: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Splits a batch of horizontally concatenated tensors back into two batches.

    Args:
        concatenated_tensor (torch.Tensor): The batch of concatenated tensors (B, C, H, W_total).
        original_widths_a (List[int]): A list of the original widths of the first set of tensors,
                                       which are used as the split points.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the two resulting tensor batches.
                                           Both batches are padded to have the same dimensions.
    """
    if concatenated_tensor.shape[0] != len(original_widths_a):
        raise ValueError("Number of tensors in batch must match the number of provided widths.")
    if concatenated_tensor.shape[0] == 0:
        return torch.empty(0), torch.empty(0)

    output_a, output_b = [], []
    for i, width_a in enumerate(original_widths_a):
        tensor = concatenated_tensor[i]
        split_point = min(width_a, tensor.shape[2]) # Ensure split point is not out of bounds

        output_a.append(tensor[:, :, :split_point])
        output_b.append(tensor[:, :, split_point:])

    # Pad tensors in each list to the same dimensions for batching
    max_h_a = max(t.shape[1] for t in output_a)
    max_w_a = max(t.shape[2] for t in output_a)
    max_h_b = max(t.shape[1] for t in output_b)
    max_w_b = max(t.shape[2] for t in output_b)
    
    padded_a = [F.pad(t, (0, max_w_a - t.shape[2], 0, max_h_a - t.shape[1])) for t in output_a]
    padded_b = [F.pad(t, (0, max_w_b - t.shape[2], 0, max_h_b - t.shape[1])) for t in output_b]

    return torch.stack(padded_a), torch.stack(padded_b)
