from keras import layers, models

total_words = 10000
embedding_size = 100
n_units = 128

text_in = layers.Input(shape=(None,))
embedding = layers.Embedding(total_words, embedding_size)(text_in)
x = layers.LSTM(n_units, return_sequences=True)(embedding)
x = layers.LSTM(n_units, return_sequences=True)(x)
probabilities = layers.Dense(n_units, activation="softmax")(x)
lstm = models.Model(text_in, probabilities)
