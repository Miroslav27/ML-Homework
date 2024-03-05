import numpy as np
import keras
import keras.datasets.imdb as imdb
from keras import layers
from keras.models import Sequential
from keras.preprocessing import sequence
from gensim.models import Word2Vec

# Create a reverse word index
max_features = 20000  # Only consider the top 30k words
maxlen = 200  # Only consider the first 300 words of each movie review
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Function to encode a sentence to integers
def encode_sentence(sentence, max_words=max_features):
    words = sentence.split()
    words = [word_index[word] + 3 for word in words if word in word_index and word_index[word] + 3 < max_words]
    return words


# Function to decode integers back to words
def decode_integers(integers):
    reverse_word_index[0] = "<PAD>"  # Padding
    reverse_word_index[1] = "<START>"  # Start of sequence
    reverse_word_index[2] = "<UNKNOWN>"  # Unknown word
    reverse_word_index[3] = "<UNUSED>"  # Unused
    words = [reverse_word_index.get(i-3, "<UNKNOWN>") for i in integers]
    return words

# Function to map words to Word2Vec vectors
def map_words_to_vectors(words, model):
    vectors = [model.wv[word] for word in words if word in model.wv]
    return np.array(vectors)


(x_train, y_train), (x_val, y_val) = imdb.load_data(
    num_words=max_features
)
sentences = [decode_integers(x) for x in x_train]  # Assuming x_train is a list of sentences
w2v_model = Word2Vec(sentences, vector_size=64, window=5, min_count=1, workers=4)

x_train = keras.utils.pad_sequences(x_train, maxlen=maxlen)
x_val = keras.utils.pad_sequences(x_val, maxlen=maxlen)

X_w2v_train = []
X_w2v_val = []


# Loop through each sequence in x_train
for sequence in x_train:
    decoded_review = decode_integers(sequence)
    word_vectors = map_words_to_vectors(decoded_review, w2v_model)
    X_w2v_train.append(word_vectors)

for sequence in x_val:
    decoded_review = decode_integers(sequence)
    word_vectors = map_words_to_vectors(decoded_review, w2v_model)
    X_w2v_val.append(word_vectors)

# Convert the list to a numpy array
X_w2v_train = np.array(X_w2v_train)
X_w2v_val = np.array(X_w2v_val)



model = Sequential([
    layers.Input(shape=(200, 64),),
    #layers.Embedding(max_features, 128),
    #layers.Bidirectional(layers.LSTM(64, return_sequences=True), backward_layer=layers.LSTM(64, activation='relu', return_sequences=True,
    #                  go_backwards=True), merge_mode="sum"),
    layers.LSTM(64, return_sequences=True),
    layers.Dropout(0.2),
    layers.LSTM(64,),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(1, activation="sigmoid")])
#model.summary()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_w2v_train, y_train, batch_size=32, epochs=5, validation_data=(X_w2v_val, y_val))

y_pred=model.predict(X_w2v_val)
"""
sentence = "I look forward for next chapter. Fun cool amazing"
encoded_sentence = keras.utils.pad_sequences(encode_sentence(sentence), maxlen=maxlen)
print(f"Sentence: {sentence}")
print(f"Encoded: {encoded_sentence}")
#print(model.predict(np.array([map_words_to_vectors(encoded_sentence,w2v_model)])))



не встиг реалізувати предикт по фразі, бо ці вектори туди сюди у слова не дуже зручно гоняти
"""