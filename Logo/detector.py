import torch
from typing import List, Tuple
from PIL import Image
from similarity import ImageMatcher
from Logo.Logo_detection import LogoBatchCropperTensor
from Logo.pasting_utils import ImagePaster
from Logo.Utils import concatenate_images_from_memory, concatenate_masks_from_memory, deconcatenate_batch_horizontally
cropTensor = LogoBatchCropperTensor()
matcher = ImageMatcher()
paster = ImagePaster()

def process_logo(
    input_tensor_ref: torch.Tensor,
    input_tensor_cand: torch.Tensor,
    prompt: str = "a logo, text, a brand crest, an emblem",
    threshold: float = 0.3,
    matching_threshold: float = 0.6,
    detection_padding_ref: int = 24,
    detection_padding_cand: int = 24,
    crop_padding_ref: int = 0,
    crop_padding_cand: int = 0
):
    candidate_images, candidate_masks, candidate_blank_masks, candidate_bboxes = cropTensor.process_image(
        image_tensor=input_tensor_cand,
        prompt=prompt,
        threshold=threshold,
        detection_padding=detection_padding_cand,
        crop_padding=crop_padding_cand
    )
    ref_images, _, ref_blank_masks, _ = cropTensor.process_image(
        image_tensor=input_tensor_ref,
        prompt=prompt,
        threshold=threshold,
        detection_padding=detection_padding_ref,
        crop_padding=crop_padding_ref
    )
    match_ref_images, match_ref_blank_masks,match_images, matched_masks, _, matched_bounding_boxes= matcher.find_best_matches_from_memory(
        ref_images=ref_images,
        ref_blank_masks=ref_blank_masks,
        candidate_images=candidate_images,
        candidate_masks=candidate_masks,
        candidate_blank_masks=candidate_blank_masks,
        candidate_bboxes=candidate_bboxes,
        threshold=matching_threshold
    )
    #Debugging
    print("match_ref_images",len(match_ref_images))
    print("match_images",len(match_images))
    print("candidate_masks",len(matched_masks))
    pixel_space, original_width = concatenate_images_from_memory(
        match_ref_images, match_images)
    inpaint_mask_input, _ = concatenate_masks_from_memory(
        match_ref_blank_masks, matched_masks)
    print("pixel_space shape:",pixel_space.shape)
    print("pixel_space size:",pixel_space.size())
    print("inpaint_mask",inpaint_mask_input.shape)
    for i, img in enumerate( match_ref_images):
        print("ref_img:",img.size)
    # Debugging ends
    return pixel_space, original_width, inpaint_mask_input, matched_bounding_boxes,  match_ref_images


def deconcatenation(
    base_image_tensor: torch.Tensor,
    input_tensor: torch.Tensor, 
    original_width_a: List[int],
    matched_bounding_bboxes: List[Tuple[int, int, int, int]],
    blend_amount: float = 0.25,
    sharpen_amount: int = 0
) -> torch.Tensor:

    if input_tensor.ndim != 4:
        raise ValueError(f"Input tensor must be a 4D batch tensor (B, C, H, W), but got shape {input_tensor.shape}.")
    
    match_images = []
    for single_image_tensor in input_tensor:
        if single_image_tensor.is_floating_point():
            single_image_tensor = single_image_tensor.mul(255).byte()

        image_np = single_image_tensor.cpu().permute(1, 2, 0).numpy()
        pil_image = Image.fromarray(image_np)
        match_images.append(pil_image)
    
    print(f"Converted batch tensor into a list of {len(match_images)} PIL images.")

    _, logo = deconcatenate_batch_horizontally(
        concatenated_images=match_images, original_widths_a=original_width_a
    )

    Final_Image = paster.paste_images_sequentially(
        base_image_tensor=base_image_tensor, 
        crop_images=logo, 
        bounding_boxes=matched_bounding_bboxes, 
        blend_amount=blend_amount, 
        sharpen_amount=sharpen_amount
    )
    
    return Final_Image
