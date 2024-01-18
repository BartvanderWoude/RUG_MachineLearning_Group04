import torch
import pandas as pd
from skimage import io
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CBISDDSM(Dataset):
    """CBIS-DDSM dataset."""
    def __init__(self, file="train.csv", path="", transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.path = path
        self.data = pd.read_csv(path + file)
        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize((224,224), antialias = True),
                                        transforms.Normalize((0.5,),
                                                            (0.5,))
                                        ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data.iloc[idx,0]
        image = io.imread(self.path + img_name)

        if self.transform:
            image = self.transform(image)
            # image = image.view(image.shape[-2],image.shape[-1])

        target = self.data.iloc[idx, 1]
        target = torch.from_numpy(np.array(target, dtype=int))
        target = torch.nn.functional.one_hot(target, num_classes=2).to(torch.float32)
        sample = {'image': image, 'class': target}

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