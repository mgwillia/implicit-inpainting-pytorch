import argparse
import numpy as np
import torch
from model import refine_model

from dataset import IIVIDataset
from torchvision import transforms


def train(args, model, train_dataset, optimizer):
    dataloader = torch.utils.data.DataLoader(train_dataset, num_workers=4,
            batch_size=5, pin_memory=True, drop_last=True, shuffle=True)

    model.train()
    for epoch in range(args.num_epochs):
        losses = []
        for _, batch in enumerate(dataloader):
            images = batch['image'].cuda(non_blocking=True)
            masks = batch['mask'].cuda(non_blocking=True)
            aug_masks = batch['aug_mask'].cuda(non_blocking=True)
            shift_h = torch.randint(high=aug_masks.shape[2], size=(1,))[0]
            shift_w = torch.randint(high=aug_masks.shape[3], size=(1,))[0]
            aug_masks = torch.roll(aug_masks, (shift_h, shift_w), dims=(2,3))
            prepared_masks = torch.zeros_like(masks)
            prepared_masks[(masks+aug_masks)>0] = 1.0
            masked_images = images*(1.-masks)
            print(masked_images.shape)
            print(prepared_masks.shape)
            model_input = torch.cat([masked_images, prepared_masks], dim=1)
            coarse_output, fine_output = model(model_input, prepared_masks) 
            loss = torch.mean(torch.abs(images - coarse_output)*(1-masks)) + torch.mean(torch.abs(images - fine_output)*(1-masks))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())

        print(f'loss: {np.mean(np.array(losses))}')
        if epoch % args.chkpt_freq == 0:
            torch.save(model.state_dict(), f'{args.dir_chkpt}/checkpoint_{epoch}.pth.tar')


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

    args.chkpt_freq = 20
    args.num_epochs = 2 #2000
    lr_init = 2e-4
    model = refine_model().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_init)

    train(args, model, dataset, optimizer)
    torch.save(model.state_dict(), f'{args.dir_chkpt}/checkpoint_final.pth.tar')

if __name__ == '__main__':
    main()
