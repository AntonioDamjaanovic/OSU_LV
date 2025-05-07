import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
img = Image.imread("LV7/imgs/test_1.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()

model = KMeans(n_clusters=5, init='random', random_state=0)
labels = model.fit_predict(img_array_aprox)
centroids = model.cluster_centers_

img_pred = centroids[labels]

new_img = img_pred.reshape(w, h, d)

plt.figure()
plt.title("Nova slika")
plt.imshow(new_img)
plt.tight_layout()
plt.show()

def new_pictures(file, n_clusters):
    img = Image.imread(file)
    plt.figure()
    plt.title("Originalna slika")
    plt.imshow(img)
    plt.tight_layout()
    plt.show()

    if file != 'LV7/imgs/test_4.jpg':
        img = img.astype(np.float64) / 255
    w,h,d = img.shape
    img_array = np.reshape(img, (w*h, d))
    img_array_aprox = img_array.copy()

    model = KMeans(n_clusters = n_clusters, init='random', random_state=0)
    labels = model.fit_predict(img_array_aprox)
    centroids = model.cluster_centers_

    img_pred = centroids[labels]

    new_img = img_pred.reshape(w, h, d)

    plt.figure()
    plt.title("Nova slika")
    plt.imshow(new_img)
    plt.tight_layout()
    plt.show()


#for i in range(2, 7):
    #new_pictures(f'imgs/test_{i}.jpg', 2)


K_values = range(1, 10)
J_values = []
for k in K_values:
    model = KMeans(n_clusters=k, init='random', random_state=0)
    labels = model.fit_predict(img_array_aprox)
    centroids = model.cluster_centers_

    J_values.append(model.inertia_)

plt.figure()
plt.plot(K_values, J_values)
plt.title('Optimalan broj clustera')
plt.xlabel('K')
plt.ylabel('J')
plt.show()


model = KMeans(n_clusters=5, init='random', random_state=0)
labels = model.fit_predict(img_array_aprox)
centroids = model.cluster_centers_

img_pred = centroids[labels]

gray = np.ones(img_pred.shape[0])
for label in labels:
    for i in img_pred.shape[0]:
        gray[i] = 0

    new_img = gray.reshape(w, h, d)
        
    plt.figure()
    plt.title("Binarna slika")
    plt.imshow(new_img)
    plt.tight_layout()
    plt.show()




