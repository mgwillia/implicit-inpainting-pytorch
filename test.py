import argparse
import numpy as np
import torch
import torch.nn.functional as F
from model import refine_model

from dataset import IIVIDataset
from torchvision import transforms


def eval(args, model, dataset):
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=4,
            batch_size=1, pin_memory=True, drop_last=False, shuffle=False)

    psnr_list = []
    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            images = batch['image'].cuda(non_blocking=True)
            masks = batch['mask'].cuda(non_blocking=True)
            masked_images = images*(1.-masks)
            model_input = torch.cat([masked_images, masks], dim=1)
            coarse_output, fine_output = model(model_input, masks)
            #print(f'Should be same shape, squeeze? {fine_output.shape} {images.shape}')
            l2_loss = F.mse_loss(fine_output, images, reduction='none')
            psnr = -10 * torch.log10(l2_loss.flatten(start_dim=1).mean(1) + 1e-15)
            psnr_list.append(psnr)

    print(f'PSNR: {torch.stack(psnr_list, 0).cpu().mean().item()}')


def main():
    ### SET UP PARSER ###
    parser = argparse.ArgumentParser(description='Paths for videos and masks')
    parser.add_argument('--dir-video', type=str, help='learning rate')
    parser.add_argument('--dir-mask', type=str, help='learning rate')
    parser.add_argument('--dir-chkpt', type=str, help='learning rate')
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.Resize((320,600)),
        transforms.ToTensor()])

    dataset = IIVIDataset(args.dir_video, args.dir_mask, transform=transform)

    model = refine_model().cuda()
    saved_model = torch.load(f'{args.dir_chkpt}/checkpoint_final.pth.tar', map_location='cpu')
    missing = model.load_state_dict(saved_model, strict=False)
    print(missing)
    model.cuda()
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    eval(args, model, dataset)

if __name__ == '__main__':
    main()
