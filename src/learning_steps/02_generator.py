import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, Model

noise_dim = 100
num_classes = 10
embed_dim = 50

class_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

def build_generator():
    # Two inputs: noise and label
    noise_in = layers.Input(shape=(noise_dim,))
    label_in = layers.Input(shape=(1,), dtype="int32")

    # Label -> embedding vector of 50 numbers
    label_vec = layers.Embedding(num_classes, embed_dim)(label_in)
    label_vec = layers.Flatten()(label_vec)

    # Glue noise and label together (100 + 50 = 150)
    x = layers.Concatenate()([noise_in, label_vec])

    # Dense layer, then reshape into a small 7x7 image
    x = layers.Dense(7 * 7 * 256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((7, 7, 256))(x)

    # 7x7 -> 14x14
    x = layers.Conv2DTranspose(128, 5, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    # 14x14 -> 28x28
    x = layers.Conv2DTranspose(64, 5, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    # Final image: 28x28x1, values between -1 and 1
    out = layers.Conv2DTranspose(1, 5, strides=1, padding="same", activation="tanh")(x)

    return Model([noise_in, label_in], out, name="generator")

generator = build_generator()
generator.summary()

# Sanity check: one image per class from the UNTRAINED generator
noise = tf.random.normal([10, noise_dim])
labels = tf.constant(np.arange(10).reshape(-1, 1))
fake = generator([noise, labels], training=False)

print("Output shape:", fake.shape)
print("Min/Max:", float(tf.reduce_min(fake)), float(tf.reduce_max(fake)))

plt.figure(figsize=(12, 3))
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(fake[i].numpy().squeeze(), cmap="gray")
    plt.title(class_names[i], fontsize=8)
    plt.axis("off")
plt.show()