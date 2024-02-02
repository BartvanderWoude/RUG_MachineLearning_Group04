import torch
import torchvision
from tqdm import tqdm
from torch.utils.data import DataLoader

import code.dataset as dataset
from code.transform import TransformPipeline

def augmentate(file="CBIS-DDSM/train.csv"):
    # Create file's dataset and dataloader; used to cycle through all images in file's dataset
    data = dataset.CBISDDSM(file=file, enable_preprocessing=False)
    data_loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=4)

    # Get dataset's file name and extension (e.g. "train" and "csv")
    file_name = file.rsplit( ".", 1 )[ 0 ]
    file_extension = file.rsplit( ".", 1 )[ 1 ]

    # Open a new file corresponding to the augmented dataset and write the header to this file
    augmented_csv = open(file_name + "-augmented." + file_extension, "w")
    augmented_csv.write("\"image_path\",\"class\"\n")

    # Setup the transformation pipeline for augmentation of the images
    transform = TransformPipeline()

    # Cycle through all images in the dataset and apply the transformation pipeline to each image
    for sample in tqdm(data_loader, position=0, leave=True):
        # Get the image and target from the sample
        image = transform.to_image(sample['image'][0])
        target = str(sample['class'][0].item())
        
        # Write original image_path and class to the augmented dataset file
        augmented_csv.write("\"" + sample['image_path'][0] + "\"," + target + "\n")

        image_path = sample['image_path'][0].rsplit( ".", 1 )[ 0 ]

        for aug_id, augmentation in enumerate(transform.augmentate(image, 4)):
            # Get one of the augmented images and save the image and write the image_path and class to the augmented dataset file
            image = augmentation[0]
            path = image_path + "-aug-" + str(aug_id) + ".jpg"

            torchvision.utils.save_image(image, path)
            augmented_csv.write("\"" + path + "\"," + target + "\n")
    augmented_csv.close()
            
if __name__ == '__main__':
    augmentate()