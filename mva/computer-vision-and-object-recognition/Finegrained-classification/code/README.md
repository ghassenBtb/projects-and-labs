## Object recognition and computer vision 2020/2021

### Assignment 3: Image classification 

#### Requirements
1. Install PyTorch from http://pytorch.org

2. Run the following command to install additional dependencies

```bash
pip install -r requirements.txt
```
#### Reshuffle and crop images
1. Run the scrip `shuffle_train_valid.py` to shuffle train and valid images while keeping same number of images per classe.
2. Run `crop.py` to detect and crop bird images.


#### Training and validating your model
Run the script `main.py` to train the model.

#### Evaluating your model on the test set

As the model trains, model checkpoints are saved to files such as `model_x.pth` to the current working directory.
Take one of the checkpoints and run:

```
python evaluate.py --data [data_dir] --model [model_file]
```

That generates a file `kaggle.csv` that can be uploaded to the private kaggle competition website.

#### Acknowledgments
Adapted from Rob Fergus and Soumith Chintala https://github.com/soumith/traffic-sign-detection-homework.<br/>
Adaptation done by Gul Varol: https://github.com/gulvarol
