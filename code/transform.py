from torchvision import transforms

class Logger:
    """Configurable pipeline with preprocessing and augmentation options"""
    def __init__(self):
        ##############################################
        ### Pipeline definitions for internal use ###
        ##############################################

        """A pipeline for resizing an image"""
        self._resize_pipeline = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224), antialias=True)
        ])

        """A pipeline for converting an image into a tensor"""
        self._to_tensor_pipeline = transforms.Compose([
            transforms.ToTensor()
        ])

        """A pipeline for converting a tensor into an image"""
        self._to_image_pipeline = transforms.Compose([
            transforms.ToPILImage()
        ])

        """The preprocessing pipeline. Do not use directly."""
        self._preprocess_pipeline = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224), antialias=True),
            transforms.ColorJitter(brightness=0, contrast=0.5, saturation=0, hue=0),
            transforms.Grayscale(),
            transforms.Normalize((0.5,),(0.5,))
        ])

        """The augmentation pipeline. Do not use directly."""
        self._augmentate_pipeline = transforms.Compose([
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



    #######################################
    ### Pipeline calls for external use ###
    #######################################
     
    def resize(self, image):
        """Resizes the image to 224, 224"""
        return self._resize_pipeline(image)
    
    def to_tensor(self, image):
        """Converts a PIL image to a tensor"""
        return self._to_tensor_pipeline(image)
    
    def to_image(self, tensor):
        """Converts a tensor to a PIL image"""
        return self._to_image_pipeline(tensor)

    def preprocess(self, image):
        """Preprocesses the source image"""
        return self._preprocess_pipeline(image)
    
    def augmentate(self, image, number_of_augmentations):
        """Augmentates the image generating multiple variations."""
        augmentations = []

        for _ in range(number_of_augmentations):
            augmentations.append(self._augmentate_pipeline(image))

        return augmentations
