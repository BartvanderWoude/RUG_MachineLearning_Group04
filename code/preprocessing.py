from torchvision import transforms

class PreprocessPipeline:
    """Configurable pipeline with preprocessing and augmentation options"""
    def __init__(self):
        """The preprocessing pipeline. Do not use directly."""
        self._preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224), antialias=True),
            transforms.ColorJitter(brightness=0, contrast=0.5, saturation=0, hue=0),
            transforms.Grayscale(),
            transforms.Normalize((0.5,),(0.5,))
        ])

        """The augmentation pipeline. Do not use directly."""
        self._augmentate = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224), antialias=True),
            transforms.RandomApply([
                transforms.RandomChoice([
                    transforms.RandomVerticalFlip(1),
                    transforms.RandomHorizontalFlip(1),
                ]),
            ]),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
        ])

    def preprocess(self, image):
        """Preprocesses the source image"""
        return self._preprocess(image)
    
    def augmentate(self, image, number_of_augmentations):
        """Augmentates the image generating multiple variations."""
        augmentations = []

        for _ in range(number_of_augmentations):
            augmentations.append(self._augmentate(image))

        print(len(augmentations))

        return augmentations
