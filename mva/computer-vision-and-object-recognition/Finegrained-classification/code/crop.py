import torchvision
import torch

from PIL import Image
import torchvision.transforms as transforms
import numpy as np


parser = argparse.ArgumentParser(description='RecVis A3 training script')
parser.add_argument('--data-crop', type=str, default='./data-resampled-crop/', 
                    help="folder where cropped images will be saved: the valid/train folders will be created by this scrip")
parser.add_argument('--data-original', type=str, default='./data-resampled/',
                    help="folder path to the original shuffled images")

args = parser.parse_args()

#get all train and valid image paths
img_paths = [p.replace('\\','/') for p in glob.glob(args['data-original'] + '*/*/*.jpg')]
transform = transforms.Compose([transforms.ToTensor()])

#Load the pretrained fasterrcnn pretrained model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()


#detect bird images, crop the bounding box and save
for (i,img_path) in enumerate(img_paths):
    new_img_path = img_path.replace(args['data-original'],args['data-crop'])
    
    if not os.path.exists(new_img_path):
        #open and process original image
        img = Image.open(img_path)
        img1 = transform(img)
        #predict with fasterrcnn, return the coordinate prediction with highest probability
        #that it is at first position (it must be a bird)
        pred = model([img1])[0]
        
        #only crop if high prediction score and class label is a bird
        if (pred['labels'][0] == 16) and (pred['scores'][0] >0.8):
            bbox = pred['boxes'][0].detach().numpy()
            bbox = tuple(bbox.tolist())

            if not os.path.exists('/'.join(new_img_path.split('/')[:-1])):
                os.mkdir('/'.join(new_img_path.split('/')[:-1]))
            img.crop(bbox).save(new_img_path)
        else:
            if not os.path.exists('/'.join(new_img_path.split('/')[:-1])):
                os.mkdir('/'.join(new_img_path.split('/')[:-1]))
            shutil.copy(img_path, new_img_path)
        if i%10==0:
            print("done : ",i)