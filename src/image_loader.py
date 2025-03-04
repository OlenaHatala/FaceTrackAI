import cv2
import os
import matplotlib.pyplot as plt

def load_image(path):
    """
    Loads an image from the given file path,
    converts it from BGR to RGB format, and returns the image.
    """
    img = cv2.imread(path)
    if img is None:
        print(f"File {path} is not found.")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb


def show_image(img, title='Image'):
    """
    Displays the given image using Matplotlib with an optional title.
    """
    plt.imshow(img)
    plt.title(title)
    # plt.axes('off')
    plt.show()


def load_images_from_dir(dir_path):
    """
    Loads all images from the specified directory, 
    converts them to RGB format, and returns them as a list of tuples.
    """
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