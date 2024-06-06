import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

model = load_model('/SOC-AI/my_model.hdf5')
tokenizer = Tokenizer()

with open("/SOC-AI/The_Verdict.txt", "r", encoding = 'utf-8') as f:
    text = f.read()

index = text.find("I HAD always")
text = text[index:]
index2 = text.find("\n\n\n\n\n\nThis work is in the public")
text = text[:index2]
text = text.lower()

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

my_input_sequence = []
for line in text.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        my_input_sequence.append(n_gram_sequence)
        
max_sequence_len = max([len(x) for x in my_input_sequence])

st.title("Next Word Predictor")
input_text = st.text_input("Enter your sentence")
next_words = st.number_input("Number of next predicted words",1,10)

def prediction(input_text,next_words,max_sequence_len,token_list):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([input_text])[0]
        print(token_list)
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        input_text += " " + output_word
    return input_text

output_text = prediction(input_text,next_words,max_sequence_len,token_list)

st.write(output_text)
