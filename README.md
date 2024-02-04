# RUG MachineLearning Group04
## Setup
Download the Kaggle dataset from https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset. Unzip and make sure the folder structure is:
```
analysis/
code/
test.py
CBIS-DDSM/
CBIS-DDSM/train.csv
CBIS-DDSM/train-augmented.csv
```
 
Where CBIS-DDSM contains a csv/ and jpeg/ folder. After unzipping the dataset folder, move to the analysis/ folder and run:
```
Rscript get_traincsv.R
```

To get the augmented dataset, move back to the main folder and run:
```
python augmentation.py
```

Note: steps are required to be done in order!

## Usage
In order to train the model, run:
```
python train_model.py
```

In order to test the model, run:
```
python test_model.py
```

## Workflow
Create your own branch for development to which you can push freely. When merging with main create a pull request from your own branch. 
