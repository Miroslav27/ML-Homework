import numpy as np
import keras
import keras.datasets.imdb as imdb
import keras_nlp
#import torch
import os
#os.environ["KERAS_BACKEND"] = "torch"  # Or "jax" or "torch"
# Create a reverse word index
max_features = 20000  # Only consider the top 30k words
maxlen = 200  # Only consider the first 300 words of each movie review
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(torch.cuda.is_available(),device)
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

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


(x_train, y_train), (x_val, y_val) = imdb.load_data(
    num_words=max_features
)
x_train = keras.utils.pad_sequences(x_train, maxlen=maxlen)
x_val = keras.utils.pad_sequences(x_val, maxlen=maxlen)
x_train_sentences = []
x_val_sentences = []
# Loop through each sequence in x_train making integers string again
for sequence in x_train:
    decoded_review = " ".join(decode_integers(sequence))
    x_train_sentences.append(decoded_review)
for sequence in x_val:
    decoded_review = " ".join(decode_integers(sequence))
    x_val_sentences.append(decoded_review)

x_train_sentences=np.array(x_train_sentences)
x_val_sentences=np.array(x_val_sentences)

# Pretrained classifier.
model = keras_nlp.models.DistilBertClassifier.from_preset(
    "distil_bert_base_en_uncased",
    num_classes=2,
)
model.backbone.trainable = False
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(1e-4),
    jit_compile=True,
)
model.summary()
#model.to(device)
model.fit(x=x_train_sentences, y=y_train, batch_size=32)
# Test Predictions
y_pred = model.predict(x_val_sentences)
print(y_pred[42], x_val_sentences[42])

sentence = "I look forward for the next chapter awesome movie"
print(f"Sentence: {sentence}", model.predict([sentence])[0])
"""
"""