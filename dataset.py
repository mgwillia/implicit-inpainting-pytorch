import os
import torch
from PIL import Image


class IIVIDataset(torch.utils.data.Dataset):
    def __init__(self, images_path, masks_path, transform=None):
        super(IIVIDataset, self).__init__()
        self.cur_folders = []
        self.transform = transform

        self.img_paths = []
        self.mask_paths = []
        num_images = len(os.listdir(masks_path))
        for i in range(num_images):
            self.img_paths.append(os.path.join(images_path, f'{str(i).zfill(5)}.jpg'))
            self.mask_paths.append(os.path.join(masks_path, f'{str(i).zfill(5)}.png'))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        image_path = self.img_paths[index]
        mask_path = self.mask_paths[index]
        aug_mask_path = self.mask_paths[torch.randint(high=len(self.imgs), size=(1,))[1]]
        with open(image_path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        with open(mask_path, 'rb') as f:
            mask = Image.open(f).convert('L')
        with open(aug_mask_path, 'rb') as f:
            aug_mask = Image.open(f).convert('L')

        if self.transform is not None:
            img = self.transform(img)
            mask = self.transform(mask)
            aug_mask = self.transform(aug_mask)

        out = {'image': img, 'mask': mask, 'aug_mask': aug_mask}

        return out
