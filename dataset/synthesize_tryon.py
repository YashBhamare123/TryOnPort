from PIL import Image
from dwpose import DwposeDetector
import torch
from torchvision.transforms import ToTensor, ToPILImage
import collections
from pydantic import BaseModel

from modules.image_pre import resize_image, grow_and_blur_mask 
from modules.segformer import SegmentCategories, FashionLabels, create_masks_subject, create_masks_garment


Output = collections.namedtuple('Output', ['image', 'mask', 'control', 'garment'])
class PreprocessConfig(BaseModel):
    resized_height : int
    resized_width : int = 0
    keep_ratio : bool = True
    resize_mode : str = 'bilinear'
    crop_padding : int = 16
    grow_padding : int = 20

class Processor():
    def __init__(self, params : PreprocessConfig):
        self.dw_model = DwposeDetector.from_pretrained_default()
        self.params = params

    def __call__(self, subject: Image.Image, garment: Image.Image) -> Output:
        sub_img = ToTensor()(subject).unsqueeze(0)
        gar_img = ToTensor()(garment).unsqueeze(0)
        pose, _, _ = self.dw_model(
            subject,
            include_hand=True,
            include_face=True,
            include_body=True,
            image_and_json=True,
            detect_resolution=1024,
        )
        pose_img = ToTensor()(pose).unsqueeze(0)

        sub_img = resize_image(sub_img, self.params.resized_height, self.params.resized_width, self.params.keep_ratio, self.params.resize_mode)
        gar_img = resize_image(gar_img, self.params.resized_height, self.params.resized_width, self.params.keep_ratio, self.params.resize_mode)
        pose_img = resize_image(pose_img, self.params.resized_height, self.params.resized_width, self.params.keep_ratio, self.params.resize_mode)

        labels_sub = SegmentCategories(
            upper_clothes = True,
            dress = True,
            lower_neck = True,
            left_arm = True,
            right_arm = True
        )
        labels_gar = FashionLabels(
            unlabelled= False
        )

        sub_mask = create_masks_subject(sub_img, labels_sub)
        gar_mask = create_masks_garment(gar_img, labels_gar)

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
        
        gar_img[:, :, (gar_mask == 0.)[0, 0, :, :]] = 1.
        sub_mask = grow_and_blur_mask(sub_mask, self.params.grow_padding)

        print(gar_img.size())
        print(gar_mask.size())
        print(sub_img.size())
        print(sub_mask.size())
        B, _, H, W = gar_img.size()
        sub_width = sub_img.size()[-1]

        blank_mask = torch.zeros((B, 1, H, W))
        black_RGB_mask = torch.zeros((B, 3, H, W))
        inpaint_img = torch.concat([sub_img, gar_img], dim = -1)
        inpaint_mask = torch.concat([sub_mask, blank_mask], dim = -1)
        pose_img = torch.concat([pose_img, black_RGB_mask], dim = -1)

        _, _, H1, W1 = inpaint_img.size()
        _, _, H2, W2 = inpaint_mask.size()

        if (H1, W1) != (H2, W2):
            raise Exception('Height and Width of final image and final mask must match')

        image = ToPILImage()(inpaint_img[0])
        mask = ToPILImage()(inpaint_mask[0])
        pose = ToPILImage()(pose_img[0])
        gar = ToPILImage()(gar_img[0])


        return Output(image, mask, pose, gar)


if __name__ == "__main__":
    params = PreprocessConfig(
        resized_height= 1024,
        grow_padding= 40,
    )
    processor = Processor(params = params)
    subject = Image.open('images/test/subject.jpg')
    garment = Image.open('images/test/garment.webp')

    output = processor(subject, garment)

    output.image.save('outputs/image.png')
    output.mask.save('outputs/mask.png')
    output.control.save('outputs/control.png')
    output.garment.save('outputs/garment.png')

