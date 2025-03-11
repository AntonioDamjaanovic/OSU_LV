import numpy as np
import matplotlib.pyplot as plt

def main():
    black = np.zeros((50, 50))
    white = np.ones((50, 50))
    upper = np.hstack((black, white))
    lower = np.hstack((white, black))

    img = np.vstack((upper, lower))
    
    plt.figure()
    plt.imshow(img, cmap="gray")
    plt.show()

if __name__ == "__main__":
    main()