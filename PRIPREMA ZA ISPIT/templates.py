import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import max_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import sklearn.linear_model as lm
from sklearn . preprocessing import OneHotEncoder
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans
import numpy as np
from tensorflow import keras
from keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import numpy as np
from tensorflow import keras
from keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from keras import saving
from PIL import Image
import numpy as np
from tensorflow import keras
from keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from keras import saving
import numpy as np
import tensorflow
from tensorflow import keras
from keras import layers, utils, datasets
#from keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from matplotlib import pyplot as plt
from tensorflow.keras.utils import to_categorical

# LOADING THE DATA
data = pd.read_csv('')
X = data
y = data[1]
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.2, random_state=1)
img = Image.imread("LV7/imgs/test_1.jpg")

# EDITING DATA
print(data.isnull().sum())              # broj izostalih vrijednosti
data = data.drop(columns=['sex'])       # brisanje stupca
data.dropna(axis=0, inplace=True)       # brisanje redova s izostalim vrijednostima
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))
img_array_aprox = img_array.copy()

# SCALING DATASETS
scaler = MinMaxScaler()
X_train_n = scaler.fit_transform(X_train)
X_test_n = scaler.transform(X_test)
scaler = StandardScaler()

# OneHotEncoder
ohe = OneHotEncoder()
X_encoded = ohe.fit_transform(data[['Fuel Type']]).toarray()

# MANUALNO KODIRANJE
data['species'].replace({'Adelie' : 0,
                        'Chinstrap' : 1,
                        'Gentoo': 2}, inplace = True)

# LINEARNA REGRESIJA
linearModel = lm.LinearRegression()
linearModel.fit(X_train, y_train)
y_test_pred = linearModel.predict(X_test)           # predan je skaliran skup
print("Model Coefficients:", linearModel.coef_)     # koeficjenti modela

# CALCULATING ERRORS
MAE = mean_absolute_error(y_test, y_test_pred)
max_err = max_error(y_test, y_test_pred)

# BINARNA LOGISTIČKA REGRESIJA
LogRegression_model = LogisticRegression()
LogRegression_model.fit(X_train, y_train)
theta_1, theta_2 = LogRegression_model.coef_[0]
theta_0 = LogRegression_model.intercept_[0]
y_train_p = LogRegression_model.predict(X_train_n)
y_test_p = LogRegression_model.predict(X_test_n)
conf_matr = confusion_matrix(y_test, y_test_pred)
display = ConfusionMatrixDisplay(confusion_matrix(y_test, y_test_pred))
display.plot()
plt.show()
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))

# VIŠEKLASNA LOGISTIČKA REGRESIJA
model = LogisticRegression(multi_class='ovr', max_iter=200)         # OvR metoda
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=['Adelie', 'Chinstrap', 'Gentoo'])

# K NAJBLIŽIH SUSJEDA KNN
KNN_model = KNeighborsClassifier(n_neighbors=5)
KNN_model.fit(X_train_n, y_train)
y_train_p_KNN = KNN_model.predict(X_train_n)
y_test_p_KNN = KNN_model.predict(X_test_n)
print("\nKNN klasifikacija: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_KNN))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p_KNN))))

# UNAKRSNA VALIDACIJA KNN
k_values = [i for i in range (1,30)]
scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X_train_n, y_train, cv=5)
    scores.append(np.mean(score))

sns.lineplot(x = k_values, y = scores, marker = 'o')
plt.xlabel("K Values")
plt.ylabel("Accuracy Score")
plt.show()
print(f'best k param is: {scores.index(max(scores)) + 1} with cross_val_score: {max(scores)}')


# SVM MODEL
SVM_model = svm.SVC(kernel='rbf', gamma=1, C=0.1)
SVM_model.fit(X_train_n, y_train)
y_train_p_SVM = SVM_model.predict(X_train_n)
y_test_p_SVM = SVM_model.predict(X_test_n)
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_SVM))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p_SVM))))

# UNAKRSNA VALIDACIJA SVM
param_grid = {
    'C': [1, 10, 100, 100],
    'gamma': [10, 1, 0.1, 0.01]
    }
svm_gscv = GridSearchCV(estimator=svm.SVC(kernel='rbf'), param_grid = param_grid, cv=5, scoring='accuracy', n_jobs=-1)
svm_gscv.fit(X_train_n, y_train)
print('\n')
print(svm_gscv.best_params_['C'])
print(svm_gscv.best_params_['gamma'])
print(svm_gscv.best_score_)

# ALGORITAM GRUPIRANJA K SREDNJIH VRIJEDNOSTI
new_points = 'testni skup'
model = KMeans(n_clusters=3, init='random', random_state=0)
model.fit(X)
labels = model.predict(new_points)
centroids = model.cluster_centers_
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]
labels = model.fit_predict(img_array_aprox)
img_pred = centroids[labels]
new_img = img_pred.reshape(w, h, d)

# METODA LAKTA
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

# UMJETNE NEURONSKE MREŽE
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))                 # karakteristike
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))                    # karakteristike
num_classes = 10
y_train_s = keras.utils.to_categorical(y_train, num_classes)
y_test_s = keras.utils.to_categorical(y_test, num_classes)
model = keras.Sequential()
model.add(layers.Input(shape = (784,)))
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Dense(50, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))
model.summary()
model.compile(
    loss='categorical_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy',]
    )
x_train_2 = x_train.reshape(60000, 784)
x_test_2 = x_test.reshape(10000, 784)
batch_size = 32
epochs = 5
history = model.fit(
    x_train_2, 
    y_train_s, 
    batch_size = batch_size, 
    epochs = epochs, 
    validation_split = 0.1
    )
predictions = model.predict(x_test_2)
score = model.evaluate(x_test_2, y_test_s, verbose=0)
print('\nTest loss:', score[0])
print('Test accuracy:', score[1])
predictions_indices = np.argmax(predictions, axis=1)
display = ConfusionMatrixDisplay(confusion_matrix(y_test, predictions_indices))
display.plot()
plt.show()
model.save("neuro.keras")
model = saving.load_model('neuro.keras')

# KONVOLUCIJSKE NEURONSKE MREŽE
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
model = keras.Sequential()
model.add(layers.Input(shape=(32,32,3)))
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(500, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(10, activation='softmax'))
model.summary()
my_callbacks = [
    keras.callbacks.TensorBoard(log_dir = 'logs/cnn_dropout',
                                update_freq = 100),
    keras.callbacks.EarlyStopping(monitor="val_loss",
                                  patience = 5,
                                  verbose = 1)
]

model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
model.fit(X_train,
            y_train,
            epochs = 10,
            batch_size = 64,
            callbacks = my_callbacks,
            validation_split = 0.1)
score = model.evaluate(X_train_n, y_train, verbose=0)
print(f'Tocnost na trening skupu podataka: {100.0*score[1]:.2f}')
score = model.evaluate(X_test_n, y_test, verbose=0)
print(f'Tocnost na testnom skupu podataka: {100.0*score[1]:.2f}')