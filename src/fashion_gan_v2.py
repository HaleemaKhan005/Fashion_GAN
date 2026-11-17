import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from keras import layers, Model

import matplotlib
matplotlib.use("Agg")   # save figures to files, no pop-up windows
import matplotlib.pyplot as plt

# ---------------- Settings ----------------
noise_dim = 100
num_classes = 10
embed_dim = 50
batch_size = 128
EPOCHS = 30         # run this file on Colab (GPU), not on the laptop
cols = 8            # images per class in the progress grid

class_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# ---------------- Folders (separate from the baseline) ----------------
BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "outputs_v2"
MODEL_DIR = BASE_DIR / "models_v2"
OUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# ---------------- Data ----------------
(x_train, y_train), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
x_train = (x_train.astype("float32") - 127.5) / 127.5
x_train = np.expand_dims(x_train, axis=-1)
y_train = y_train.astype("int32").reshape(-1, 1)

dataset = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
           .shuffle(len(x_train))
           .batch(batch_size))

# ---------------- Generator (unchanged) ----------------
def build_generator():
    noise_in = layers.Input(shape=(noise_dim,))
    label_in = layers.Input(shape=(1,), dtype="int32")

    label_vec = layers.Embedding(num_classes, embed_dim)(label_in)
    label_vec = layers.Flatten()(label_vec)
    x = layers.Concatenate()([noise_in, label_vec])

    x = layers.Dense(7 * 7 * 256, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((7, 7, 256))(x)

    x = layers.Conv2DTranspose(128, 5, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(64, 5, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    out = layers.Conv2DTranspose(1, 5, strides=1, padding="same", activation="tanh")(x)
    return Model([noise_in, label_in], out, name="generator")

# ---------------- Discriminator (CHANGED: stricter teacher) ----------------
def build_discriminator():
    img_in = layers.Input(shape=(28, 28, 1))
    label_in = layers.Input(shape=(1,), dtype="int32")

    label_vec = layers.Embedding(num_classes, embed_dim)(label_in)
    label_vec = layers.Flatten()(label_vec)
    label_vec = layers.Dense(28 * 28)(label_vec)
    label_img = layers.Reshape((28, 28, 1))(label_vec)

    x = layers.Concatenate()([img_in, label_img])

    # 28x28 -> 14x14
    x = layers.Conv2D(64, 5, strides=2, padding="same")(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.2)(x)          # CHANGED: 0.3 -> 0.2

    # 14x14 -> 7x7
    x = layers.Conv2D(128, 5, strides=2, padding="same")(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.2)(x)          # CHANGED: 0.3 -> 0.2

    # 7x7 -> 4x4   (NEW extra layer)
    x = layers.Conv2D(256, 5, strides=2, padding="same")(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Flatten()(x)
    out = layers.Dense(1)(x)
    return Model([img_in, label_in], out, name="discriminator")

generator = build_generator()
discriminator = build_discriminator()

# ---------------- Losses and optimizers (unchanged) ----------------
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# ---------------- One training step (unchanged) ----------------
@tf.function
def train_step(real_images, real_labels):
    batch = tf.shape(real_images)[0]
    noise = tf.random.normal([batch, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        fake_images = generator([noise, real_labels], training=True)

        real_output = discriminator([real_images, real_labels], training=True)
        fake_output = discriminator([fake_images, real_labels], training=True)

        g_loss = generator_loss(fake_output)
        d_loss = discriminator_loss(real_output, fake_output)

    g_grads = gen_tape.gradient(g_loss, generator.trainable_variables)
    d_grads = disc_tape.gradient(d_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))

    return g_loss, d_loss

# ---------------- Fixed samples to watch progress ----------------
seed_noise = tf.random.normal([num_classes * cols, noise_dim])
seed_labels = tf.constant(np.repeat(np.arange(num_classes), cols).reshape(-1, 1),
                          dtype=tf.int32)

def save_image_grid(epoch):
    images = generator([seed_noise, seed_labels], training=False).numpy()
    fig, axes = plt.subplots(num_classes, cols, figsize=(cols * 1.2, num_classes * 1.2))
    for i in range(num_classes):
        for j in range(cols):
            ax = axes[i, j]
            ax.imshow(images[i * cols + j].squeeze(), cmap="gray", vmin=-1, vmax=1)
            ax.axis("off")
            if j == 0:
                ax.text(-4, 14, class_names[i], ha="right", va="center", fontsize=8)
    fig.suptitle(f"Epoch {epoch}")
    fig.savefig(OUT_DIR / f"epoch_{epoch:03d}.png", bbox_inches="tight", dpi=100)
    plt.close(fig)

# ---------------- Training loop ----------------
def train(dataset, epochs):
    g_history, d_history = [], []
    save_image_grid(0)   # the "before training" picture

    for epoch in range(1, epochs + 1):
        start = time.time()
        g_losses, d_losses = [], []

        for images, labels in dataset:
            g_loss, d_loss = train_step(images, labels)
            g_losses.append(g_loss.numpy())
            d_losses.append(d_loss.numpy())

        g_avg, d_avg = float(np.mean(g_losses)), float(np.mean(d_losses))
        g_history.append(g_avg)
        d_history.append(d_avg)

        save_image_grid(epoch)
        print(f"Epoch {epoch:3d}/{epochs} | Gen loss: {g_avg:.4f} | "
              f"Disc loss: {d_avg:.4f} | {time.time() - start:.0f}s")

    np.save(OUT_DIR / "g_loss.npy", np.array(g_history))
    np.save(OUT_DIR / "d_loss.npy", np.array(d_history))
    generator.save(MODEL_DIR / "generator.keras")
    print("Done. Images are in outputs_v2/, generator is saved in models_v2/.")

if __name__ == "__main__":
    train(dataset, EPOCHS)