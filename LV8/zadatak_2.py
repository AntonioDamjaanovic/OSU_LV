import numpy as np
from tensorflow import keras
from keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from keras import saving

model = saving.load_model('neuro.keras')
model.summary()

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_test_2 = x_test.reshape(10000, 784)
predictions = model.predict(x_test_2)
y_pred = np.argmax(predictions, axis = 1)

for i in range(50):
    if y_test[i] != y_pred[i]:
        
        plt.imshow(x_test[i])
        plt.title(f'Predicted number: {y_pred[i]}')
        plt.show()
        
