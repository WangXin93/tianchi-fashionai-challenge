# Flip all images in a folder
import os
import sys
from PIL import Image
 
 
def flip_image(image_path, saved_location):
    """
    Flip or mirror the image
 
    @param image_path: The path to the image to edit
    @param saved_location: Path to save the cropped image
    """
    image_obj = Image.open(image_path)
    rotated_image = image_obj.transpose(Image.FLIP_LEFT_RIGHT)
    rotated_image.save(saved_location)
 

if __name__ == '__main__':
    root_dir = sys.argv[-1]
    for root, dirs, files in os.walk(root_dir):
        for name in files:
            img = os.path.join(root, name)
            print(img)
            flip_image(img, img)
