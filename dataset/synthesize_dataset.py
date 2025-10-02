import PIL
from datasets import load_dataset, Dataset, Image
from ultralytics import YOLO
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage, ToTensor, Resize
import torchvision
from tqdm import tqdm

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_list, mask_list):
        super().__init__()  
        self.img_list = img_list
        self.mask_list = mask_list
    
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        transform = torchvision.transforms.Compose([
            ToTensor(),
            Resize(size = (640, 640))
        ])
        return (transform(self.img_list[idx]),transform(self.mask_list[idx]), self.img_list[idx].size)
    
class YoloModify:
    def __init__(self):
        self.model = YOLO('models/yolov8n-face.pt')

    def get_largest_index(self, element):
        current_index = -1
        current_box = -1
        for idx, box in enumerate(element):
            if (box[2]-box[0]) * (box[3] - box[1]) > current_box:
                    current_index = idx
                    current_box = box
    
        return current_index

    def grow_mask(self, mask : torch.Tensor, padding : int) -> torch.Tensor:
        mask = F.max_pool2d(
            input = mask,
            kernel_size = 2*padding + 1,
            stride = 1,
            padding= (2 * padding + 1) // 2,
        )
        return mask


    def modify_mask(self, images: torch.Tensor, masks :torch.Tensor, size: tuple, mask_padding :int, class_label = 18.):
        new_masks = []
        new_images = []
        image = images
        mask = masks
        results = self.model(image, verbose = False)

        if results[0].boxes.xywh.size(0) > 0 and len(results) > 0:
            for idx, result in enumerate(results):
                if len(result) > 0:
                    lar_index = self.get_largest_index(result[0].boxes.xywh)
                    box_coors = result.boxes.xyxy[lar_index]
                    _, w, h = image[idx].size()
                    yolo_mask = torch.zeros(1, h, w)
                    yolo_mask[:, int(box_coors[1]):int(box_coors[3]), int(box_coors[0]):int(box_coors[2])] = 11.
                    yolo_mask = self.grow_mask(yolo_mask, mask_padding)   
                    cur_mask = mask[idx]
                    cur_mask[(cur_mask == 11. / 255.) & (yolo_mask == 0.)] = class_label / 255.
                    w = size[0][idx]
                    h = size[1][idx]
                    cur_mask = cur_mask.unsqueeze(0)
                    print(cur_mask.size())
                    cur_mask = F.interpolate(
                        input = cur_mask,
                        size= (h, w),
                        mode = 'bilinear',
                        align_corners= False,
                    ).squeeze(0)
                    cur_img = F.interpolate(
                        input = image[idx].unsqueeze(0),
                        size = (h, w),
                        mode = 'bilinear',
                        align_corners = False
                    ).squeeze(0)
                    pil_mask = ToPILImage(mode = 'L')(cur_mask)
                    pil_image = ToPILImage(mode = 'RGB')(cur_img)
                    new_masks.append(pil_mask)
                    new_images.append(pil_image)
        return new_images, new_masks  


def main():
    ds = load_dataset('mattmdjaga/human_parsing_dataset', split = 'train')
    splits = np.arange(1, 18, 4)
    mask_output = []
    image_output = []
    for start, end in tqdm(zip(splits[:-1], splits[1:])):

        s_ds = ds[start:end]
        dataset = ImageDataset(img_list = s_ds['image'], mask_list= s_ds['mask'])
        dataloader = DataLoader(dataset, batch_size = 4, num_workers = 0)
        
        output_dataset = []
        image_dataset = []
        model = YoloModify()
        for images, masks, size in dataloader:
            out_images, out_masks = model.modify_mask(images, masks, size, mask_padding = 10)
            output_dataset = out_masks + output_dataset
            image_dataset  = out_images + image_dataset
        mask_output = mask_output + output_dataset
        image_output = image_output + image_dataset

    dataset_new = Dataset.from_dict({"image" : image_output, "mask" : mask_output})
    dataset_new.push_to_hub('YashBhamare123/human_parsing_dataset_plus_neck')

if __name__ == "__main__":
    main()