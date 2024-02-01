import torch
import torchvision
from tqdm import tqdm
from torch.utils.data import DataLoader

import code.dataset as dataset
import code.model as model
import code.train as train
from code.transform import TransformPipeline

def augmentate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    data = dataset.CBISDDSM(enable_preprocessing=False)
    data_loader = DataLoader(data, batch_size=1, shuffle=True, num_workers=4)

    transform = TransformPipeline()

    for sample in tqdm(data_loader, position=0, leave=True):
        image = transform.to_image(sample['image'][0])
        image_name = sample['image_name'][0].rsplit( ".", 1 )[ 0 ]

        aug_id = 0
        for augmentation in transform.augmentate(image, 4):
            image = augmentation[0]
            path = image_name + "-aug-" + str(aug_id) + ".jpg"
            aug_id += 1

            torchvision.utils.save_image(image, path)
            # print("Saved augmentation to ", path)
            
if __name__ == '__main__':
    augmentate()