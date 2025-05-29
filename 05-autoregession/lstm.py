import json
import re
import string

import kagglehub
import numpy as np
import tensorflow as tf
from keras import callbacks, layers, losses, models

# Load data
# path = kagglehub.dataset_download("hugodarwood/epirecipes")

with open("./data/epireceipes/versions/2/full_format_receipes.json") as json_data:
    recipe_data = json.load(json_data)

filtered_data = [
    "Recipe for " + x["title"] + "|" + " ".join(x["directions"])
    for x in recipe_data
    if "title" in x
    and x["title"] is not None
    and "directions" in x
    and x["directions"] is not None
]


def pad_punctuation(s):
    s = re.sub(f"[{string.punctuation}]", r" \1 ", s)
    s = re.sub(" +", " ", s)
    return s


text_data = [pad_punctuation(x) for x in filtered_data]
text_ds = tf.data.Dataset.from_tensor_slices(text_data).batch(32).shuffle(1000)

vectorize_layer = layers.TextVectorization(
    standardize="lower",
    max_tokens=10000,
    output_mode="int",
    output_sequence_length=200 + 1,
)

vectorize_layer.adapt(text_ds)
vocab = vectorize_layer.get_vocabulary()


def prepare_inputs(text):
    text = tf.expand_dims(text, -1)
    tokenized_sentences = vectorize_layer(text)
    x = tokenized_sentences[:, :-1]
    y = tokenized_sentences[:, 1:]
    return x, y


train_ds = text_ds.map(prepare_inputs)


class TextGenerator(callbacks.Callback):
    def __init__(self, index_to_word, top_k=10):
        self.index_to_word = index_to_word
        self.word_to_index = {word: index for index, word in enumerate(index_to_word)}

    def sample_from(self, probs, temperature):
        probs = probs ** (1 / temperature)
        probs = probs / np.sum(probs)
        return np.random.choice(len(probs), p=probs), probs

    def generate(self, start_prompt, max_tokens, temperature):
        start_tokens = [self.word_to_index.get(x, 1) for x in start_prompt.split()]
        sample_token = None  # Initialize sample_token
        info = []
        while len(start_tokens) < max_tokens and sample_token != 0:
            x = np.array([start_tokens])
            y = self.model.predict(x, verbose=0)  # Add verbose=0 to reduce output
            sample_token, probs = self.sample_from(y[0][-1], temperature)
            info.append({"prompt": start_prompt, "word_probs": probs})
            start_tokens.append(sample_token)
            start_prompt = start_prompt + " " + self.index_to_word[sample_token]
        print(f"\nGenerated text:\n{start_prompt}\n")
        return info

    def on_epoch_end(self, epoch, logs=None):
        self.generate("recipe for", max_tokens=100, temperature=1.0)


inputs = layers.Input(shape=(None,), dtype="int32")
x = layers.Embedding(10000, 100)(inputs)
x = layers.LSTM(128, return_sequences=True)(x)
outputs = layers.Dense(10000, activation="softmax")(x)
lstm = models.Model(inputs, outputs)

loss_fn = losses.SparseCategoricalCrossentropy()
lstm.compile("adam", loss_fn)

text_generator = TextGenerator(vocab)

lstm.fit(train_ds, epochs=25, callbacks=[text_generator])
