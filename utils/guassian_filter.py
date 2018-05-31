import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import gaussian_filter


def main(): #filename, bbox
    filename = './Img/img/img_00000001.jpg'
    img = plt.imread(filename)

    #x_1     53
    #y_1    130
    #x_2    289
    #y_2    440
    bbox = [53, 130, 289, 440]
    mask = np.zeros(img.shape)
    mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
    # Gaussian filter
    mask = gaussian_filter(mask, sigma=min(img.shape[:2])/10)

    blur = mask * img

    #plt.imshow(img, cmap='gray')
    plt.imshow(blur.astype(np.uint8))
    plt.show()

if __name__ == "__main__":
    main()
