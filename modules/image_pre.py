from PIL import Image
from torchvision.transforms import PILToTensor, ToPILImage, GaussianBlur
import requests
import torch
import torch.nn.functional as F
from pydantic import BaseModel

from segformer import create_masks, SegmentCategories

class InpaintStitch(BaseModel, arbitrary_types_allowed=True):
    original : list[torch.Tensor]
    cropped : list[torch.Tensor]
    coords : list

def load_image_from_url(url : str) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Loads an image from a URL as a PyTorch tensor and returns its RGB tensor and optional alpha mask.

    Args:
        url (str): Image URL.

    Returns:
        Tuple[torch.Tensor, torch.Tensor | None]: 
            RGB tensor of shape [1, C, H, W], and alpha mask tensor ([1, 1, H, W]) if present, else None.
    
    """

    img = Image.open(requests.get(url, stream= True).raw)
    mask = None
    img_rgb = img.convert('RGB')
    ts = PILToTensor()(img_rgb).unsqueeze(0) / 255.

    if 'A' in img.getbands():
        mask = PILToTensor()(img.getchannel('A')).unsqueeze(0)/ 255.
        print(mask.size())
    elif img.mode == 'P' and 'transparency' in img.info:
        mask = PILToTensor()(img.convert('RGBA').getchannel('A')).unsqueeze(0)/ 255.
    
    if mask:
        mask = mask.unsqueeze(0)
    
    return (ts, mask)


def resize_image(img : torch.Tensor, h : int, w : int = 0, keep_ratio = True, mode = 'bilinear') -> torch.Tensor:
    if keep_ratio:
        w = int((h / img.size()[-2]) * img.size()[-1])
    
    if w == 0:
        raise Exception("Width has to be non-zero while resizing")

    img = F.interpolate(
        input = img,
        size = (h, w),
        mode = mode,
        align_corners= False
    )

    return img


def crop_mask(img : torch.Tensor, mask : torch.Tensor, padding = 0):
    _, _, height, width = img.size()
    
    non_zero_id = torch.nonzero(mask.squeeze(0))
    print(non_zero_id.size())

    x1 = torch.clamp(torch.min(non_zero_id[:, 2]) - padding, min = 0)
    x2 = torch.clamp(torch.max(non_zero_id[:, 2]) + padding, max = width -1)
    y1 = torch.clamp(torch.min(non_zero_id[:, 1]) - padding, min = 0)
    y2 = torch.clamp(torch.max(non_zero_id[:, 1]) + padding, max = height -1)

    crop_img = img[:, :, y1:y2 +1 , x1:x2 +1]
    crop_mask = mask[:, :, y1:y2 +1 , x1:x2 +1]
    
    out = InpaintStitch(original= [img, mask], cropped = [crop_img, crop_mask], coords= [(x1, y1), (x2, y2)])
    return out


def grow_and_blur_mask(mask : torch.Tensor, padding= 0):
    mask = F.max_pool2d(mask, kernel_size= padding+ 1)
    blur = GaussianBlur(kernel_size= 5, sigma = 0.5)
    mask = blur(mask)
    return mask



if __name__ == "__main__":
    img, _ = load_image_from_url(url = 'https://plus.unsplash.com/premium_photo-1673210886161-bfcc40f54d1f?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8MXx8cGVyc29uJTIwc3RhbmRpbmd8ZW58MHx8MHx8&w=1000&q=80')
    seg_opts = SegmentCategories(upper_clothes= True, face=True)
    print(img.size())
    mask = create_masks(img, seg_opts)
    out = crop_mask(img, mask, padding = 10)
    
    cropped_image = out.cropped[0]
    pil = ToPILImage()(cropped_image[0])
    pil.show()
    



    
