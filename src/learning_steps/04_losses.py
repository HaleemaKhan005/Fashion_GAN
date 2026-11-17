import tensorflow as tf

# The loss: BCE on raw scores (logits)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)    # real -> target 1
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)   # fake -> target 0
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)         # wants fakes to look real

# Two separate optimizers
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)

# ---- Sanity checks with made-up scores ----

# Case 1: discriminator completely unsure (score 0 means 50/50)
unsure = tf.zeros([8, 1])
print("Case 1 - unsure discriminator")
print("  Disc loss:", float(discriminator_loss(unsure, unsure)))
print("  Gen loss: ", float(generator_loss(unsure)))

# Case 2: discriminator is winning (real scores high, fake scores low)
real_scores = tf.fill([8, 1], 5.0)
fake_scores = tf.fill([8, 1], -5.0)
print("Case 2 - discriminator winning")
print("  Disc loss:", float(discriminator_loss(real_scores, fake_scores)))
print("  Gen loss: ", float(generator_loss(fake_scores)))

# Case 3: generator is winning (discriminator thinks fakes are real)
fooled_fake = tf.fill([8, 1], 5.0)
print("Case 3 - generator winning")
print("  Gen loss: ", float(generator_loss(fooled_fake)))