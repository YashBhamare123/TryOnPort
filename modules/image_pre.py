from PIL import Image, ImageOps
from torchvision.transforms import PILToTensor, ToPILImage, GaussianBlur
import requests
import torch
import torch.nn.functional as F
from pydantic import BaseModel
from io import BytesIO

from .segformer import create_masks_subject, create_masks_garment, SegmentCategories, FashionLabels
from .config import PreprocessConfig
from .utils import show_tensor
from .intellisegment import get_segments

class InpaintStitch(BaseModel, arbitrary_types_allowed=True):
    original : tuple[torch.Tensor, torch.Tensor]
    cropped : tuple[torch.Tensor, torch.Tensor]
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

    headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    }

    img = Image.open(BytesIO(requests.get(url, headers= headers).content))
    exif_data = img.getexif()


    if exif_data:
        img = ImageOps.exif_transpose(img)
    
    mask = None
    img_rgb = img.convert('RGB')
    ts = PILToTensor()(img_rgb).unsqueeze(0) / 255.

    if 'A' in img.getbands():
        mask = PILToTensor()(img.getchannel('A')).unsqueeze(0)/ 255.
        print(mask.size())
    elif img.mode == 'P' and 'transparency' in img.info:
        mask = PILToTensor()(img.convert('RGBA').getchannel('A')).unsqueeze(0)/ 255.
        ts = PILToTensor()(img.convert('RGBA').convert('RGB')).unsqueeze(0) / 255.
    
    if mask is not None:
        mask = mask.unsqueeze(0)
    
    return (ts, mask)


def resize_image(img : torch.Tensor, h : int, w : int = 0, keep_ratio = True, mode = 'bilinear') -> torch.Tensor:
    """
    Resizes an image tensor to specified dimensions.
    
    Args:
        img (torch.Tensor): Input image tensor of shape [B, C, H, W].
        h (int): Target height.
        w (int, optional): Target width. If 0 and keep_ratio=True, calculated from height. Defaults to 0.
        keep_ratio (bool, optional): Whether to maintain aspect ratio. Defaults to True.
        mode (str, optional): Interpolation mode. Defaults to 'bilinear'.
    
    Returns:
        torch.Tensor: Resized image tensor.
    """
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

#TODO Add stitch function

def crop_mask(img : torch.Tensor, mask : torch.Tensor, padding = 0):
    """
    Crops image and mask to the bounding box of non-zero mask values with optional padding.
    
    Args:
        img (torch.Tensor): Input image tensor of shape [B, C, H, W].
        mask (torch.Tensor): Binary mask tensor of shape [B, 1, H, W].
        padding (int, optional): Additional padding around the crop area. Defaults to 0.
    
    Returns:
        InpaintStitch: Object containing original tensors, cropped tensors, and crop coordinates.
    """
    _, _, height, width = img.size()
    
    non_zero_id = torch.nonzero(mask.squeeze(0))

    if non_zero_id.numel() == 0:
        raise Exception('Cannot Crop on Empty Mask')
    
    print(non_zero_id[:, 2].size())

    x1 = torch.clamp(torch.min(non_zero_id[:, 2]) - padding, min = 0)
    x2 = torch.clamp(torch.max(non_zero_id[:, 2]) + padding, max = width -1)
    y1 = torch.clamp(torch.min(non_zero_id[:, 1]) - padding, min = 0)
    y2 = torch.clamp(torch.max(non_zero_id[:, 1]) + padding, max = height -1)

    crop_img = img[:, :, y1:y2 +1 , x1:x2 +1]
    crop_mask = mask[:, :, y1:y2 +1 , x1:x2 +1]
    
    out = InpaintStitch(original= (img, mask), cropped = (crop_img, crop_mask), coords= [(x1, y1), (x2, y2)])
    return out


def grow_and_blur_mask(mask : torch.Tensor, padding= 0):
    """
    Grows a mask using max pooling and applies Gaussian blur for smooth edges.
    
    Args:
        mask (torch.Tensor): Input binary mask tensor.
        padding (int, optional): Amount to grow the mask. Defaults to 0.
    
    Returns:
        torch.Tensor: Processed mask with grown and blurred edges.
    """
    mask = F.max_pool2d(mask, kernel_size=2*padding + 1, stride=1, padding= (2*padding + 1 )// 2)
    blur = GaussianBlur(kernel_size= 5, sigma = 0.5)
    mask = blur(mask)
    return mask

class PreprocessImage:
    def __init__(self, params : PreprocessConfig):
        self.params = params
    
    def preprocess(self, subject_url : str, garment_url : str):
        sub = load_image_from_url(subject_url)
        sub_img = sub[0]

        # show_tensor(sub_img, 'after load')

        sub_trans_mask = sub[1]
        gar_img = load_image_from_url(garment_url)[0]

        sub_img = resize_image(sub_img, self.params.resized_height, self.params.resized_width, self.params.keep_ratio, self.params.resize_mode)
        gar_img = resize_image(gar_img, self.params.resized_height, self.params.resized_width, self.params.keep_ratio, self.params.resize_mode)

        # show_tensor(sub_img, 'after resize')

        inputs = get_segments(subject_url, garment_url)
        print(inputs)
        labels_sub = SegmentCategories(**inputs)
        if labels_sub.upper_clothes or labels_sub.dress:
            labels_sub.lower_neck = True

        labels_gar = FashionLabels(unlabelled= False)

        sub_mask = create_masks_subject(sub_img, labels_sub)
        gar_mask = create_masks_garment(gar_img, labels_gar)

        # sub_crop = crop_mask(sub_img, sub_mask, padding = self.params.crop_padding)
        # gar_crop = crop_mask(gar_img, gar_mask, padding = self.params.crop_padding)
        # sub_img, sub_mask = sub_crop.cropped
        # gar_img, gar_mask = gar_crop.cropped

        # show_tensor(sub_img, 'After Crop')
        sub_img = resize_image(sub_img, 
                               self.params.resized_height, 
                               self.params.resized_width, 
                               self.params.keep_ratio, 
                               self.params.resize_mode)
        sub_mask = resize_image(sub_mask, 
                                self.params.resized_height, 
                                self.params.resized_width, 
                                self.params.keep_ratio, 
                                self.params.resize_mode)
        gar_img = resize_image(gar_img, 
                               self.params.resized_height, 
                               self.params.resized_width, 
                               self.params.keep_ratio, 
                               self.params.resize_mode)
        gar_mask = resize_image(gar_mask, 
                                self.params.resized_height, 
                                self.params.resized_width, 
                                self.params.keep_ratio, 
                                self.params.resize_mode)
        
        #TODO Replace this with transluent fill
        gar_img[:, :, (gar_mask == 0.)[0, 0, :, :]] = 1.
        sub_mask = grow_and_blur_mask(sub_mask, self.params.grow_padding)

        # show_tensor(gar_img, 'after fill')
        print(gar_img.size())
        print(gar_mask.size())
        print(sub_img.size())
        print(sub_mask.size())
        B, _, H, W = gar_img.size()


        blank_mask = torch.zeros((B, 1, H, W))
        inpaint_img = torch.concat([sub_img, gar_img], dim = -1)
        inpaint_mask = torch.concat([sub_mask, blank_mask], dim = -1)

        _, _, H1, W1 = inpaint_img.size()
        _, _, H2, W2 = inpaint_mask.size()

        if (H1, W1) != (H2, W2):
            raise Exception('Height and Width of final image and final mask must match')

        show_tensor(inpaint_img, 'inpaing_image')
        show_tensor(inpaint_mask, 'inpaint_mask')

        return inpaint_img, inpaint_mask



if __name__ == "__main__":
    params = PreprocessConfig(
        resized_height= 1024,
        keep_ratio= True,
        crop_padding= 16,
        grow_padding= 20
    )

    subject_url = "https://res.cloudinary.com/dukgi26uv/image/upload/v1759842454/the-nude-v-neck-pointelle-knit-tee-tops-snhkxv_2048x_bfnch4.webp"
    garment_url = "https://res.cloudinary.com/dukgi26uv/image/upload/v1759842480/Manchester_United_Home_92.94_Shirt_kyajol.webp"

    ts, _ = load_image_from_url(garment_url)
    show_tensor(ts)

    processor = PreprocessImage(params)
    out = processor.preprocess(subject_url, garment_url)
    show_tensor(out[0])
    show_tensor(out[1])

    



   