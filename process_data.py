"""Process data."""
import os
import numpy as np
from skimage.draw import polygon
import matplotlib.pyplot as plt
from PIL import Image


def get_images_name(image_dir):
    """function to get all the images in a directory

    Args:
        image_dir (path): path to the image directory

    Returns:
        a list of images in the image directory
    """
    images_name_list = []
    files = os.listdir(image_dir)
    for file in files:
        if file.endswith('.tif'):
            images_name_list.append(file)
    return images_name_list


def collect_segmentation(json_data, images_name_list):
    """Collect all the segmentation annotations from the json file

    Args:
        json_data (json): json file
        images_name_list (list): a list contains all the images name

    Returns:
        dictionary: a dictionary grouping all the segmentations of the dataset
    """
    segmentation_dict = {}
    for image in json_data["images"]:
        name = image["file_name"]
        if name in images_name_list:
            image_id = image["id"]
            segmentation_list = []
            for ann in json_data["annotations"]:
                if ann["image_id"] == image_id:
                    segmentation_list += ann["segmentation"]
            segmentation_dict[image_id] = {}
            segmentation_dict[image_id]["segmentation"] = segmentation_list
            segmentation_dict[image_id]["name"] = name
    return segmentation_dict


def create_mask(segmentation_dict, save_folder):
    """create an binary mask for each image

    Args:
        segmentation_dict (dict): the dictionary where the segmentations and
        the image id are stored

        save_folder (string): destination of the mask
    """
    # Images have shape (520, 704) and the segmentation contains value such as
    # 520.0 or 704.0 so here is one way to create a mask
    shape = (521, 705, 3)
    for image_id in segmentation_dict.keys():
        combined_mask = np.zeros(shape, dtype=np.float32)
        segmentation_list = segmentation_dict[image_id]["segmentation"]
        for segmentation in segmentation_list:
            # For each cell, a polygon will be drawn with the white foreground,
            # so that later we combine all the polygon to a black backgroundW
            mask = np.zeros(shape, np.bool8)
            x_cood, y_cood = polygon(segmentation[1::2], segmentation[0::2])
            mask[x_cood, y_cood] = 1
            combined_mask = np.maximum(mask, combined_mask)
        file_name = segmentation_dict[image_id]["name"]
        name = os.path.splitext(file_name)[0]
        # Save the combined mask
        plt.imsave(save_folder + name + ".png", combined_mask, cmap="gray")


def convert_grayscale(original_folder, save_folder, json_file):
    """convert imag to gray scale .png

    Args:
        original_folder (string): path to the folder of original dataset
        save_folder (string): path to the folder of destination
        json_file (json): json file
    """
    file_name_list = []
    for image in json_file["images"]:
        file_name = image["file_name"]
        name = os.path.splitext(file_name)[0]
        file_name_list.append(name)
    for image in os.listdir(original_folder):
        name = os.path.splitext(image)[0]
        if name in file_name_list:
            img_path = os.path.join(original_folder, image)
            image = Image.open(img_path)
            image = image.convert("L")
            image.save(f"{save_folder}/{name}.png")


# if __name__ == "__main__":
#     """Example for val dataset."""
#     with open("val.json", "r", encoding="utf-8") as f:
#         val_data = json.load(f)
#     images_name_list = get_images_name("test_image/val_images")
#     segmentation_dict = collect_segmentation(val_data, images_name_list)
#     create_mask(segmentation_dict, "test_image/val_masks/")
#     convert_grayscale("test_image/val_images",
#                       "test_image/val_images", val_data)
