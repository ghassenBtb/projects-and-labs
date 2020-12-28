import zipfile
import os

import torchvision.transforms as transforms

# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 64 x 64 in size
# and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from
# the training set

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



