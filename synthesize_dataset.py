import PIL
from datasets import load_dataset
from ultralytics import YOLO
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage, ToTensor, Resize
import torchvision

class ImageDataset(Dataset):
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
        return (transform(self.img_list[idx]), transform(self.mask_list[idx]))
    
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
            padding= (2* padding + 1) // 2,
        )
        return mask


    def modify_mask(self, images: torch.Tensor, masks :torch.Tensor, mask_padding :int, class_label = 18.):
        new_masks = []
        image = images
        mask = masks
        results = self.model(image)

        if results[0].boxes.xywh.size(0) > 0:
            for idx, result in enumerate(results):
                lar_index = self.get_largest_index(result[0].boxes.xywh)
                box_coors = result.boxes.xyxy[lar_index]
                _, w, h = image[idx].size()
                yolo_mask = torch.zeros(1, h, w)
                yolo_mask[:, int(box_coors[1]):int(box_coors[3]), int(box_coors[0]):int(box_coors[2])] = 11.
                yolo_mask = self.grow_mask(yolo_mask, mask_padding)   
                cur_mask = mask[idx]
                cur_mask[(cur_mask == 11. / 255.) & (yolo_mask == 0.)] = 1.
                pil = ToPILImage(mode = 'L')(cur_mask)
                new_masks.append(pil)
        return new_masks  


def main():
    ds = load_dataset('mattmdjaga/human_parsing_dataset', split = 'train')
    s_ds = ds[:10]

    mask = s_ds['mask'][0]
    mask = ToTensor()(mask)
    uniques = torch.unique(mask)

    dataset = ImageDataset(img_list = s_ds['image'], mask_list= s_ds['mask'])
    dataloader = DataLoader(dataset, batch_size = 4, shuffle = True, num_workers = 0, drop_last = True)
    
    output_dataset = []
    model = YoloModify()
    for images, masks in dataloader:
        out_masks = model.modify_mask(images, masks, mask_padding = 10)
        output_dataset = out_masks + output_dataset

    output_dataset[7].save('sample_output.png')


if __name__ == "__main__":
    main()