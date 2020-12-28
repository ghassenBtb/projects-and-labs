import zipfile
import os

import torchvision.transforms as transforms


class White_noise():
	"""
	Add a gaussian noise to image
	"""
	def __init__(self, level=0.1):
		self.level = level
	
	def __call__(self, img):
		return img+torch.randn_like(img)*(self.level*np.random.rand())


data_transforms_pretrained = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(300),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4),
        transforms.Lambda(White_noise(level = 0.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(300),
        transforms.CenterCrop(300),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}



