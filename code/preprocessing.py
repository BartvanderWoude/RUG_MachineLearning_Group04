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

    def preprocess(self, image):
        """Preprocesses the source image"""
        return self._preprocess(image)
