import os
import cv2
import matplotlib.pyplot as plt
import random

class DatasetVisualizer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.image_files = [f for f in os.listdir(dataset_path) if f.endswith(".jpg") or f.endswith(".jpeg")]

    def visualize_dataset(self, num_samples=5):
        # Randomly choose samples
        sample_files = random.sample(self.image_files, num_samples)

        # Set up subplots
        fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))

        # Display sample images
        for i, image_file in enumerate(sample_files):
            image_path = os.path.join(self.dataset_path, image_file)
            image = cv2.imread(image_path)

            # Convert BGR to RGB for displaying with Matplotlib
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Display the image
            axes[i].imshow(image_rgb)
            axes[i].set_title(image_file)
            axes[i].axis("off")

        plt.show()


