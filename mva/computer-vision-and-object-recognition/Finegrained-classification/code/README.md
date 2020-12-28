## Fine-Grained bird species classification

### Code

#### Install dependencies

```bash
pip install -r requirements.txt
```

#### Reshuffle and crop images
Reshuffle the train and valid images. The number of images per class are kept the same.
```bash
python shuffle_train_valid.py
```

Detect and crop bird images with fasterRCNN.
```bash
python crop.py
```

#### Train
Train the model. As the model trains, model checkpoints are saved to files such as `model_x.pth` to the current working directory.
```bash
python main.py
```

#### Test


Choose one of the checkpoints and run:

```
python evaluate.py --data [data_dir] --model [model_file]
```

This generates a file `kaggle.csv` that can be uploaded to the private kaggle competition website. After the upload
a test accuracy score will be computed. 


