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


class FashionLabels(BaseModel):
    unlabelled: bool = True
    shirt_blouse: bool = True
    top_t_shirt_sweatshirt: bool = True
    sweater: bool = True
    cardigan: bool = True
    jacket: bool = True
    vest: bool = True
    pants: bool = True
    shorts: bool = True
    skirt: bool = True
    coat: bool = True
    dress: bool = True
    jumpsuit: bool = True
    cape: bool = True
    glasses: bool = True
    hat: bool = True
    headband_head_covering_hair_accessory: bool = True
    tie: bool = True
    glove: bool = True
    watch: bool = True
    belt: bool = True
    leg_warmer: bool = True
    tights_stockings: bool = True
    sock: bool = True
    shoe: bool = True
    bag_wallet: bool = True
    scarf: bool = True
    umbrella: bool = True
    hood: bool = True
    collar: bool = True
    lapel: bool = True
    epaulette: bool = True
    sleeve: bool = True
    pocket: bool = True
    neckline: bool = True
    buckle: bool = True
    zipper: bool = True
    applique: bool = True
    bead: bool = True
    bow: bool = True
    flower: bool = True
    fringe: bool = True
    ribbon: bool = True
    rivet: bool = True
    ruffle: bool = True
    sequin: bool = True
    tassel: bool = True

def create_masks_subject(img : torch.Tensor, labels: SegmentCategories) -> torch.Tensor:

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

def create_masks_garment(img: torch.Tensor, labels: FashionLabels) -> torch.Tensor:
    img = img[0]
    img = ToPILImage()(img)

    processor = SegformerImageProcessor.from_pretrained('sayeed99/segformer-b3-fashion')
    model = AutoModelForSemanticSegmentation.from_pretrained('sayeed99/segformer-b3-fashion')

    inps = processor(img, return_tensors = 'pt')
    out = model(**inps)
    logits = out.logits.detach().cpu()
    upsampled_logits = nn.functional.interpolate(
        input = logits,
        size = img.size[::-1],
        mode = 'bilinear',
        align_corners= False
    )
    pred_seg = torch.argmax(upsampled_logits, dim = 1)[0]

    cats_on = [i for i, v in enumerate(labels.model_dump().values()) if v]
    cats_off = [i for i, v in enumerate(labels.model_dump().values()) if not v]

    for i in cats_on:
        pred_seg[pred_seg == i] = 255.
    
    for i in cats_off:
        pred_seg[pred_seg == i] = 0.
    
    pred_seg = pred_seg/ 255.
    

    return pred_seg.unsqueeze(0).unsqueeze(0)


if __name__ == "__main__":
    url = 'https://res.cloudinary.com/dukgi26uv/image/upload/v1754143626/tryon-images/v7mzzq0rivocfutzc57e.jpg'
    img = Image.open(requests.get(url, stream = True).raw)
    img = PILToTensor()(img).unsqueeze(0)
    labels = SegmentCategories(upper_clothes= True, pants = True)
    labels_gar = FashionLabels(unlabelled= False)
    mask = create_masks_garment(img, labels_gar)
    mask = ToPILImage()(mask[0])
    mask.save('file.png')








