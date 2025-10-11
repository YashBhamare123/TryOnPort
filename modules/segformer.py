from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from torchvision.transforms import ToPILImage, PILToTensor
import requests
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt
from pydantic import BaseModel
import torch



class SegmentCategories(BaseModel):
    background: bool = False
    hat: bool = False
    hair: bool = False
    sunglasses: bool = False
    upper_clothes: bool = False
    skirt: bool = False
    pants: bool = False
    dress: bool = False
    belt: bool = False
    left_shoe: bool = False
    right_shoe: bool = False
    face: bool = False
    left_leg: bool = False
    right_leg: bool = False
    left_arm: bool = False
    right_arm: bool = False
    bag: bool = False
    scarf: bool = False
    lower_neck : bool = False


def create_masks(img : torch.Tensor, labels: SegmentCategories) -> torch.Tensor:

    """
    Generates a binary mask tensor for a given input image based on specified segment categories using a pre-trained SegFormer semantic segmentation model.

    Args:
        img (torch.Tensor): The input image tensor of shape (1, C, H, W), typically with values in [0, 1].
        labels (SegmentCategories): An object representing category activation states. Categories marked "on" will be masked as 1, others as 0.

    Returns:
        torch.Tensor: A mask tensor of shape (1, 1, H, W) where each pixel is set to 1 if it belongs to any of the "on" categories, and 0 otherwise.
    """
    img = img[0]
    pil_img = ToPILImage()(img)

    processor = SegformerImageProcessor.from_pretrained('YashBhamare123/segformer_finetune', subfolder = 'segformer_b2_clothes_epoch_13')
    model = AutoModelForSemanticSegmentation.from_pretrained('YashBhamare123/segformer_finetune', subfolder = 'segformer_b2_clothes_epoch_13')
    inputs = processor(pil_img, return_tensors = 'pt')
    out = model(**inputs)

    logits =out.logits.detach().cpu()
    upsampled_logits = nn.functional.interpolate(
        logits,
        size=pil_img.size[::-1],
        mode="bilinear",
        align_corners=False,
    )


    pred_seg = upsampled_logits.argmax(dim=1)[0]

    cats_on = [i for i, v in enumerate(labels.model_dump().values()) if v]
    cats_off = [i for i, v in enumerate(labels.model_dump().values()) if not v]

    for i in cats_on:
        pred_seg[pred_seg == i] = 255.
    
    for i in cats_off:
        pred_seg[pred_seg == i] = 0.
    
    pred_seg = pred_seg/ 255.
    

    return pred_seg.unsqueeze(0).unsqueeze(0)

    
if __name__ == "__main__":
    url = 'https://plus.unsplash.com/premium_photo-1673210886161-bfcc40f54d1f?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8MXx8cGVyc29uJTIwc3RhbmRpbmd8ZW58MHx8MHx8&w=1000&q=80'
    img = Image.open(requests.get(url, stream = True).raw)
    img = PILToTensor()(img).unsqueeze(0)
    labels = SegmentCategories(upper_clothes= True, pants = True)
    mask = create_masks(img, labels)
    plt.imshow(mask[0][0])
    plt.show()








