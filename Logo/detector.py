
import torch
from typing import List, Tuple
from Logo.similarity_test import ImageMatcher
from Logo.Logo_detection import ProperLogo
from Logo.pasting import ImagePaster
from Logo.Utils import deconcatenate_tensors, concatenate_tensors
from torchvision.transforms import ToPILImage
# Initialize the main components of the logo processing pipeline.
# These are loaded once to be reused for multiple processing calls.
cropTensor = ProperLogo()
matcher = ImageMatcher()
paster = ImagePaster()


def process_logo(
    input_tensor_ref: torch.Tensor,
    input_tensor_cand: torch.Tensor,
    prompt: str = "a logo, text, a brand crest, an emblem",
    threshold: float = 0.3,
    matching_threshold: float = 0.6,
    detection_padding_ref: int = 24,
    detection_padding_cand: int = 0,
    crop_padding_ref: int = 0,
    crop_padding_cand: int = 24
) -> Tuple[torch.Tensor, List[int], torch.Tensor, List[List[int]], torch.Tensor]:
    """
    Processes reference and candidate images to find and match logos, preparing them for further processing.

    This function takes a reference image and a candidate image, detects potential logos in both,
    finds the best matches between them based on visual similarity, and then concatenates the
    matched pairs into a single tensor for tasks like inpainting or style transfer.

    Args:
        input_tensor_ref (torch.Tensor): The reference image tensor with shape (1, C, H, W) and dtype bfloat16.
        input_tensor_cand (torch.Tensor): The candidate image tensor with shape (1, C, H, W) and dtype bfloat16.
        prompt (str): The text prompt to guide the logo detection model.
        threshold (float): The confidence threshold for the object detection model.
        matching_threshold (float): The cosine similarity threshold for matching detected logos.
        detection_padding_ref (int): Padding to add around detected bounding boxes in the reference image.
        detection_padding_cand (int): Padding to add around detected bounding boxes in the candidate image.
        crop_padding_ref (int): Padding to add when cropping detected logos from the reference image.
        crop_padding_cand (int): Padding to add when cropping detected logos from the candidate image.

    Returns:
        Tuple[torch.Tensor, List[int], torch.Tensor, List[List[int]], torch.Tensor]: A tuple containing:
            - pixel_space (torch.Tensor): A tensor of shape (N, C, H, W_new) containing the horizontally
              concatenated pairs of matched reference and candidate logos.
            - original_widths (List[int]): A list of the original widths of the reference logos before concatenation.
            - inpaint_mask_input (torch.Tensor): A tensor of shape (N, 1, H, W_new) containing concatenated masks
              for inpainting, where the reference part is blank and the candidate part is the detected logo mask.
            - matched_bounding_boxes (List[List[int]]): The bounding boxes of the matched logos in the
              original candidate image.
            - match_ref_batch (torch.Tensor): A batch tensor of the matched reference logos.
    """
    # Detect logos in the candidate image
    candidate_images, candidate_masks, candidate_blank_masks, candidate_bboxes = cropTensor.process_image(
        image=input_tensor_cand,
        prompt=prompt,
        threshold=threshold,
        detection_padding=detection_padding_cand,
        crop_padding=crop_padding_cand
    )
    # Detect logos in the reference image
    ref_images, _, ref_blank_masks, _ = cropTensor.process_image(
        image=input_tensor_ref,
        prompt=prompt,
        threshold=threshold,
        detection_padding=detection_padding_ref,
        crop_padding=crop_padding_ref
    )

    # Find the best matches between the detected logos from reference and candidate images
    match_ref_images, match_ref_blank_masks, match_images, matched_masks, _, matched_bounding_boxes = matcher.find_best_matches(
        ref_images=ref_images,
        ref_blank_masks=ref_blank_masks,
        candidate_images=candidate_images,
        candidate_masks=candidate_masks,
        candidate_blank_masks=candidate_blank_masks,
        candidate_bboxes=candidate_bboxes,
        threshold=matching_threshold
    )

    if match_ref_images.shape[0] == 0:
        print("Warning: No matching logos found. Returning empty tensors.")
        return torch.empty(0), [], torch.empty(0), [], torch.empty(0)
    for i, img in enumerate(match_ref_images):
        ToPILImage()(img.to(torch.float32)).save(f"match_ref_image{i}.png")

    print("match_ref_images",len(match_ref_images))
    print("match_images",len(match_images))
    print("candidate_masks",len(matched_masks))
    # Concatenate the matched image pairs and mask pairs into single tensors
    pixel_space, original_widths = concatenate_tensors(
        match_ref_images, match_images)
    for i, img in enumerate(pixel_space):
        ToPILImage()(img.to(torch.float32)).save(f"pixel_space{i}.png")
    inpaint_mask_input, _ = concatenate_tensors(
        match_ref_blank_masks, matched_masks)
    for i, img in enumerate(inpaint_mask_input):
        ToPILImage()(img.to(torch.float32)).save(f"inpaint_mask{i}.png")

    return pixel_space, original_widths, inpaint_mask_input, matched_bounding_boxes, match_ref_images


def deconcatenation(
    base_image_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    original_width_a: List[int],
    matched_bounding_boxes: List[List[int]],
    blend_amount: float = 0.25,
    sharpen_amount: int = 0
) -> torch.Tensor:
    """
    Deconcatenates processed images, pastes the logos onto the base image, and returns the final image.

    This function takes a batch of horizontally concatenated images, splits them back into two halves,
    and then pastes the second half (the processed logos) onto a base image at the specified
    bounding box locations.

    Args:
        base_image_tensor (torch.Tensor): The base image tensor of shape (1, C, H, W) to paste logos onto.
        input_tensor (torch.Tensor): A batch of concatenated image tensors of shape (N, C, H, W_new)
                                     from the processing step.
        original_width_a (List[int]): A list of the original widths of the first set of images
                                      before concatenation, used as split points.
        matched_bounding_boxes (List[List[int]]): A list of bounding boxes [[x1, y1, x2, y2], ...]
                                                    specifying where to paste the logos on the base image.
        blend_amount (float): The amount of blending to apply at the edges of the pasted images.
        sharpen_amount (int): The number of times to apply a sharpening filter to the logos before pasting.

    Returns:
        torch.Tensor: The final composed image as a tensor of shape (1, C, H, W).
    """
    if input_tensor.ndim != 4:
        raise ValueError(
            f"Input tensor must be a 4D batch tensor (B, C, H, W), but got shape {input_tensor.shape}.")

    # Split the processed tensor back into the reference and the modified logo parts
    _, logo_tensors = deconcatenate_tensors(
        concatenated_tensor=input_tensor, original_widths_a=original_width_a
    )

    # Paste the processed logos onto the original candidate image
    final_image = paster.paste_images_sequentially(
        base_image_tensor=base_image_tensor,
        crop_tensors=logo_tensors,
        bounding_boxes=matched_bounding_boxes,
        blend_amount=blend_amount,
        sharpen_amount=sharpen_amount
    )

    return final_image
