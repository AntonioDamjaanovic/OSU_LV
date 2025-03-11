import numpy as np
import matplotlib.pyplot as plt

def brighten_image(img, value):
    img = img.astype(np.float64)
    img += value
    img = np.clip(img, 0, 255)

    return img

def show_img(img):
    plt.figure()
    plt.imshow(img, cmap='gray')
    plt.show()

def cut_img(img):
    img = img[:, 160:320]           # svaki red od 160-320 stupca
    show_img(img)

def rotate_img(img):
    img = np.rot90(img, axes=(1,0))
    show_img(img)

def mirror_img(img):
    img = np.flip(img, axis=(1))
    show_img(img)

def main():
    img = plt.imread('road.jpg')
    img = img[:,:,0].copy()
    brighter_img = brighten_image(img, 100)

    show_img(img)
    show_img(brighter_img)
    cut_img(img)
    rotate_img(img)
    mirror_img(img)

if __name__ == '__main__':
    main()