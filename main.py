#!/usr/bin/env python
# coding: utf-8

import os
import PIL
import torch
import wandb
import random
import datetime

from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import resnet18

from tqdm import tqdm
from dataclasses import dataclass, asdict

from typing import List


@dataclass
class Config:
    batch_size: int
    encoder_input_size: int
    momentum: float
    temperature: float
    queue_size: int
    feature_dim: int
    
    mean: List[float]
    std: List[float]


class ImageDataset(Dataset):
    def __init__(self, path: str, transforms):
        super(ImageDataset).__init__()
        self.path = path
        self.filelist = os.listdir(path)
        self.transforms = transforms
    
    def __len__(self) -> int:
        return len(self.filelist)
    
    def __getitem__(self, index: int) -> torch.Tensor:
        img = PIL.Image.open(os.path.join(self.path, self.filelist[index]))
        return self.transforms(img)
    

class PairImageDataset(ImageDataset):
    def __getitem__(self, index: int) -> torch.Tensor:
        img = PIL.Image.open(os.path.join(self.path, self.filelist[index]))
        return (self.transforms(img), self.transforms(img))


class Encoder(nn.Module):
    def __init__(self, feature_dim=128):
        super(Encoder, self).__init__()
        self.feature_dim = feature_dim
        
        net = resnet18(num_classes=feature_dim, norm_layer=nn.BatchNorm2d)
        
        self.net = []
        for name, module in net.named_children():
            if name == 'conv1':
                module  = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if isinstance(module, nn.MaxPool2d):
                continue
            if isinstance(module, nn.Linear):
                self.net.append(nn.Flatten(1))
            self.net.append(module)
            
        self.net = nn.Sequential(*self.net)
    
    def forward(self, x):
        return self.net(x)


class MoCo(nn.Module):
    def __init__(self, feature_dim: int, queue_size: int, momentum: float, temperature: float):
        super(MoCo, self).__init__()
        
        self.feature_dim = feature_dim
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature
        
        self.query_encoder = Encoder(feature_dim=feature_dim)
        self.key_encoder = Encoder(feature_dim=feature_dim)
        
        for param_q, param_k in zip(self.query_encoder.parameters(), self.key_encoder.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        self.register_buffer('queue', torch.randn(feature_dim, queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
                             
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.query_encoder.parameters(), self.key_encoder.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        
        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0, f'{self.queue_size = }, {batch_size = }'
        
        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.t()
        ptr = (ptr + batch_size) % self.queue_size # move pointer
        
        self.queue_ptr[0] = ptr
        
    def contrastive_loss(self, im_q, im_k):
        q = self.query_encoder(im_q)
        q = nn.functional.normalize(q, dim=1)
        
        device = torch.device(im_k.get_device() if torch.cuda.is_available() else 'cpu')
        k = self.key_encoder(im_k)
        k = nn.functional.normalize(k, dim=1)
        
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.temperature
        
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
        loss = nn.CrossEntropyLoss().to(device)(logits, labels)
        
        return loss, q, k
        
    def forward(self, im1, im2):
        # update the key encoder
        with torch.no_grad():
            self._momentum_update_key_encoder()

        loss, _, k = self.contrastive_loss(im1, im2)
        self._dequeue_and_enqueue(k)
        return loss


class GaussianBlur:
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma
        
    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(PIL.ImageFilter.GaussianBlur(radius=sigma))
        return x


def train(model, dataloader, optimizer, epochs):
    for e in range(epochs):
        wandb.log({'epoch': e})
        total_num, total_loss = 0, 0
        train_bar = tqdm(dataloader)
        for A, B in train_bar:
            A, B = A.to(device, non_blocking=True), B.to(device, non_blocking=True)
            loss = model(A, B)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            total_num += dataloader.batch_size
            total_loss += loss.item() * dataloader.batch_size
            wandb.log({'loss': total_loss / total_num})
            train_bar.set_description(f'train epoch: [{e}/{epochs}], loss: {total_loss / total_num:.3f}')


if __name__ == "__main__":
    wandb.init()
    config = Config(batch_size=64, 
                    encoder_input_size=64,
                    queue_size=1024,
                    momentum=0.99,
                    temperature=0.1,
                    feature_dim=32,
                    mean=[0.5432602 , 0.5414511 , 0.56252354], 
                    std=[0.13111968, 0.1271442 , 0.13114768])
    wandb.config.update(asdict(config))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(config.encoder_input_size, scale=(0.2, 1)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.05)], p=0.7),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(config.mean, config.std),
    ])

    train_dataset = PairImageDataset('/home/matz/Documents/VideoChopper/out/train/', train_transforms)

    test_transforms = transforms.Compose([
        transforms.Resize((config.encoder_input_size, config.encoder_input_size)),
        transforms.ToTensor(),
        transforms.Normalize(config.mean, config.std),
    ])

    test_dataset = ImageDataset('/home/matz/Documents/VideoChopper/out/test/', test_transforms)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

    model = MoCo(queue_size=config.queue_size, 
                 momentum=config.momentum, 
                 temperature=config.temperature, 
                 feature_dim=config.feature_dim).to(device)
    wandb.watch(model)

    optimizer = torch.optim.Adam(model.parameters())
    train(model, train_loader, optimizer, epochs=10)

    torch.save({'state_dict': model.state_dict}, f'models/model_{datetime.today().strftime("%Y%m%d_%H%M")}')
