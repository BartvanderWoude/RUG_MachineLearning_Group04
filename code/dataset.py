from .transform import TransformPipeline
# from .exploratory_analysis import DatasetVisualizer

import torch
import pandas as pd
from skimage import io
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CBISDDSM(Dataset):
    def __init__(self, file="train.csv", path="", img_size=(224,224), enable_preprocessing=True):
        self.path = path
        self.data = pd.read_csv(path + file)
        self.enable_preprocessing = enable_preprocessing

        self.transform = TransformPipeline()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data.iloc[idx,0]
        image = io.imread(self.path + img_name)

        target = self.data.iloc[idx, 1]
        target = torch.from_numpy(np.array(target, dtype=int))
        # target = torch.nn.functional.one_hot(target, num_classes=2).to(torch.float32)

        if self.enable_preprocessing:
            sample = {
                'image': self.transform.preprocess(image),
                'augmentation': self.transform.augmentate(image, 4),
                'class': target
            }
        else:
            sample = {
                'image': self.transform.resize(image),
                'image_name': img_name
            }

        return sample

def test_dataset(path):
    data = CBISDDSM(file="train.csv", path=path)
    data_loader = DataLoader(data, batch_size=4, shuffle=True, num_workers=4)
    x = 0
    for sample in data_loader:
        print(sample['image'].shape, sample['class'].shape)
        x = x + 1
        if x == 5:
            break 

if __name__ == '__main__':
    path = "../"
    test_dataset(path)