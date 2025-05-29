import tensorflow as tf
import tensorflow_probability as tfp
from keras import layers, models

dist = tfp.distributions.PixelCNN(
    image_shape=(32, 32, 1),
    num_resnet=1,
    num_hierarchy=2,
    num_filters=32,
    num_logistic_mix=5,
    dropout_p=0.3,
)
dist.sample(10).numpy()

image_input = layers.Input(shape=(32, 32, 1))
log_prob = dist.log_prob(image_input)

model = models.Model(inputs=image_input, outputs=log_prob)
model.add_loss(-tf.reduce_mean(log_prob))
