# -*- coding: utf-8 -*-
"""Language Modelling Assignment.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1QUlmxzfMgR6jVOBAqyLSrdDH1u2K5_YV

SHAWN MATYANGA
R204651S **bold text**
"""

import random
import re
import numpy as np
import keras
import gensim
from keras.preprocessing.text import Tokenizer
from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


model = load_model('best_model2.h5')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def predict_next_words(model, tokenizer, text, num_words=1):
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



#user_input = input("Please type five words in Shona: ")

#Predicting the next word
predicted_words = predict_next_words(model, tokenizer, user_input, num_words=3)
print(f"The next words might be: {predicted_words}")

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
