import os
import glob
import random
import shutil


parser = argparse.ArgumentParser(description='RecVis A3 training script')
parser.add_argument('--data-original', type=str, default='./data/',
                    help="folder path to the original images")
parser.add_argument('--data-shuffled', type=str, default='./data-resampled/',
                    help="folder where shuffled images will be saved: the valid/train folders will be created by this scrip")
args = parser.parse_args()

#return number of files by subfolders
mydict = {}
for (root,dirs,files) in os.walk('./data', topdown=False):
    if len(files)>0:
        mydict[root.replace('\\','/')]=len(files)
        
#concatenate train and validation images and shuufle        
val_img_paths = [p.replace('\\','/') for p in glob.glob("./data/val_images/*/*.jpg")]
train_img_paths = [p.replace('\\','/') for p in glob.glob("./data/train_images/*/*.jpg")]
all_paths = val_img_paths + train_img_paths
random.shuffle(all_paths)

#reconstruct train and validation sets with same distribution of images per class (as in before)
dic_labels = {label:[] for label in os.listdir('./data/train_images/')}

for p in all_paths:
    dic_labels[p.split('/')[-2]].append(p)
    
for label in dic_labels.keys():
    n_valid = mydict['./data/val_images/'+label]
    for p in dic_labels[label][:n_valid]:
        if not os.path.exists("./data-resampled/val_images/"+label):
            os.mkdir("./data-resampled/val_images/"+label)
        shutil.copy(p,"./data-resampled/val_images/"+label+'/'+p.split('/')[-1])
    for p in dic_labels[label][n_valid:]:
        if not os.path.exists("./data-resampled/train_images/"+label):
            os.mkdir("./data-resampled/train_images/"+label)
        shutil.copy(p,"./data-resampled/train_images/"+label+'/'+p.split('/')[-1])