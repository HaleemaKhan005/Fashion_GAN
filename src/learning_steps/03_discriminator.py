import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

num_classes = 10
embed_dim = 50

def build_discriminator():
    # Two inputs: image and label
    img_in = layers.Input(shape=(28, 28, 1))
    label_in = layers.Input(shape=(1,), dtype="int32")

    # Label -> embedding -> 784 numbers -> a 28x28x1 "label image"
    label_vec = layers.Embedding(num_classes, embed_dim)(label_in)
    label_vec = layers.Flatten()(label_vec)
    label_vec = layers.Dense(28 * 28)(label_vec)
    label_img = layers.Reshape((28, 28, 1))(label_vec)

    # Stack image and label image: 28x28x2
    x = layers.Concatenate()([img_in, label_img])

    # 28x28 -> 14x14
    x = layers.Conv2D(64, 5, strides=2, padding="same")(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    # 14x14 -> 7x7
    x = layers.Conv2D(128, 5, strides=2, padding="same")(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.3)(x)

    # One raw score (logit): real or fake
    x = layers.Flatten()(x)
    out = layers.Dense(1)(x)

    return Model([img_in, label_in], out, name="discriminator")

discriminator = build_discriminator()
discriminator.summary()

# ---- Sanity check (this is the part that changed) ----
(x_train, y_train), (_, _) = tf.keras.datasets.fashion_mnist.load_data()

real = (x_train[:8].astype("float32") - 127.5) / 127.5
real = np.expand_dims(real, axis=-1)
real = tf.constant(real)                                            # NumPy -> tensor

labels = tf.constant(y_train[:8].reshape(-1, 1), dtype=tf.int32)    # NumPy -> tensor

random_imgs = tf.random.uniform([8, 28, 28, 1], -1, 1)

real_scores = discriminator([real, labels], training=False)
fake_scores = discriminator([random_imgs, labels], training=False)

print("Output shape:", real_scores.shape)
print("Real scores:", real_scores.numpy().squeeze().round(3))
print("Fake scores:", fake_scores.numpy().squeeze().round(3))