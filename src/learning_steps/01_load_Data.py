
# interpreator path 
# C:\Users\Administrator\Desktop\My_Projects_2k26\Fashion_GAN\venv\Scripts\python.exe
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 1. Load the data (we only need the training part)
(x_train, y_train), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
# 2. Look at the shapes and value range
print("Images shape:", x_train.shape)
print("Labels shape:", y_train.shape)
print("Min/Max pixel:", x_train.min(), x_train.max())

# 3. Scale to -1..1 and add a channel dimension
x_train = (x_train.astype("float32") - 127.5) / 127.5
x_train = np.expand_dims(x_train, axis=-1)
print("After processing:", x_train.shape, x_train.min(), x_train.max())
# 4. Show 10 sample images with their names
class_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

plt.figure(figsize=(12, 3))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(x_train[i].squeeze(), cmap="gray")
    plt.title(class_names[y_train[i]], fontsize=8)
    plt.axis("off")
plt.show()