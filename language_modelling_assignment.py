import os
import streamlit as st
import pickle
#importing and loading  all the libraries used in this assignment
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the previously saved model
model = load_model('best_model2.h5')

st.title('Next Shona Word Predictor')
#st.button("https://colab.research.google.com/drive/1wjFB3dIQ5Anr5oXmX5BJIo4Q_TtLkSP4#scrollTo=Zx7NZs1rG9gq")


with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def predict_next_word(model, tokenizer, text, num_words=1):
    """
    Predict the next set of words using the trained model.

    Args:
    - model (keras.Model): The trained model.
    - tokenizer (Tokenizer): The tokenizer object used for preprocessing.
    - text (str): The input text.
    - num_words (int): The number of words to predict.

    Returns:
    - str: The predicted words.
    """
    for _ in range(num_words):
        # Tokenize and pad the text
        sequence = tokenizer.texts_to_sequences([text])[0]
        sequence = pad_sequences([sequence], maxlen=5, padding='pre')

        # Predict the next word
        predicted_probs = model.predict(sequence, verbose=0)
        predicted = np.argmax(predicted_probs, axis=-1)

        # Convert the predicted word index to a word
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break

        # Append the predicted word to the text
        text += " " + output_word

    return ' '.join(text.split(' ')[-num_words:])

def main():
    user_input = st.text_input('Nyora manzwi mashanu')
    lst = list(user_input.split())

    if st.button("Generate"):
        if (user_input is not None and len(lst)==5):
            result = predict_next_word(model, tokenizer, user_input, num_words=1)
            st.success(result)

        else:
            st.write("Please enter five words")

if __name__ == '__main__':
    main()
