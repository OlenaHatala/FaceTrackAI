import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        print(f"File {path} is not found.")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb


def show_image(img, title='Image'):
    plt.imshow(img)
    plt.title(title)
    # plt.axes('off')
    plt.show()


def load_images_from_dir(dir_path):
    images = []

    if not os.path.exists(dir_path):
        print(f"Error: Directory {dir_path} does not exist.")
        return images
    
    for filename in os.listdir(dir_path):
        path = os.path.join(dir_path, filename)
        try:
            img = load_image(path)
            images.append((filename, img))
        except FileNotFoundError:
            print(f"Error: File {filename} could not be downloaded.")
    return images