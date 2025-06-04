import kagglehub
import tensorflow as tf
from keras import utils

# Load data
# path = kagglehub.dataset_download("nunenuh/pytorch-challange-flower-dataset")

train_data = utils.image_dataset_from_directory(
    "./data/pytorch-challenge-flower-dataset/versions/3/dataset",
    labels=None,
    image_size=(64, 64),
    batch_size=None,
    shuffle=True,
    seed=42,
    interpolation="bilinear",
)


def preprocess(img):
    img = tf.cast(img, "float32") / 255.0
    return img


train = train_data.map(lambda x: preprocess(x))
train = train.repeat(5)
train = train.batch(64, drop_remainder=True)
