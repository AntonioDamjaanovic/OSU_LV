from PIL import Image
import numpy as np
from tensorflow import keras
from keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from keras import saving

model = saving.load_model('neuro.keras')
model.summary()

def preprocess_image(img_path):
    img = Image.open(img_path).convert('L')
    img = img.resize((28, 28)) 
    
    img = Image.eval(img, lambda x: 255 - x)
    
    img_array = np.array(img).astype('float32') / 255.0
    img_array = img_array.reshape(1, 784) 
    return img_array

img_path = "test3.png"
img_array = preprocess_image(img_path)

prediction = model.predict(img_array)
predicted_class = np.argmax(prediction, axis=1)
confidence = np.max(prediction) * 100

print(f"\nPredicted digit: {predicted_class[0]} (confidence: {confidence:.2f}%)")

plt.imshow(img_array.reshape(28, 28), cmap='gray')
plt.title(f"Predicted: {predicted_class[0]}")
plt.axis('off')
plt.show()
