import os
import random
import cv2
import matplotlib.pyplot as plt

# Set the path to your dataset directory
dataset_path = "../CBIS-DDSM/jpeg/CBIS-DDSM/jpeg/1.3.6.1.4.1.9590.100.1.2.126082211045731020508108042042916052/"

# Function to load and display sample images
def display_sample_images(dataset_path, num_samples=5):
    # Get a list of image files in the dataset directory
    image_files = [f for f in os.listdir(dataset_path) if f.endswith(".jpg") or f.endswith(".jpeg")]

    # Display a random sample of images
    for _ in range(num_samples):
        # Choose a random image file
        image_file = random.choice(image_files)

        # Load the image using OpenCV
        image_path = os.path.join(dataset_path, image_file)
        image = cv2.imread(image_path)

        # Display the image using Matplotlib
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(image_file)
        plt.show()

# Function to get basic information about the images
def inspect_images(dataset_path):
    image_files = [f for f in os.listdir(dataset_path) if f.endswith(".jpg") or f.endswith(".jpeg")]

    for image_file in image_files:
        image_path = os.path.join(dataset_path, image_file)
        image = cv2.imread(image_path)

        # Print basic information
        print(f"Image: {image_file}")
        print(f"Shape: {image.shape}")
        print(f"Minimum pixel value: {image.min()}")
        print(f"Maximum pixel value: {image.max()}")
        print()

# Display sample images
display_sample_images(dataset_path)

# Inspect basic information about the images
inspect_images(dataset_path)
