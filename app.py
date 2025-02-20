import streamlit as st
import numpy as np
import pickle
import os
import tensorflow as tf

# Function to manually pad sequences without TensorFlow
def pad_sequences_custom(sequences, maxlen, padding='post', value=0):
    padded_sequences = np.full((len(sequences), maxlen), value, dtype=np.int32)
    
    for i, seq in enumerate(sequences):
        if len(seq) > maxlen:
            padded_sequences[i] = seq[:maxlen]  # Truncate if longer
        else:
            if padding == 'post':
                padded_sequences[i, :len(seq)] = seq
            else:
                padded_sequences[i, -len(seq):] = seq

    return padded_sequences

# Load saved models if they exist
encoder_path = 'encoder_model.h5'
decoder_path = 'decoder_model.h5'

if os.path.exists(encoder_path) and os.path.exists(decoder_path):
    encoder_model = tf.keras.models.load_model(encoder_path)
    decoder_model = tf.keras.models.load_model(decoder_path)
else:
    encoder_model = None
    decoder_model = None

# Load tokenizers
try:
    with open('kn_tokenizer.pkl', 'rb') as f:
        kn_tokenizer = pickle.load(f)

    with open('en_tokenizer.pkl', 'rb') as f:
        en_tokenizer = pickle.load(f)
except Exception as e:
    st.error(f"Error loading tokenizers: {e}")
    kn_tokenizer = None
    en_tokenizer = None

# Set max sequence lengths
if encoder_model and decoder_model:
    max_kn_length = encoder_model.input_shape[1]
    max_en_length = decoder_model.input_shape[0][1]
else:
    max_kn_length = 20  # Default length
    max_en_length = 20

# Function to translate Kannada text to English
def translate_sentence(input_text):
    if not encoder_model or not decoder_model:
        return "Model not found. Please upload the trained model."

    if not kn_tokenizer or not en_tokenizer:
        return "Tokenizer not found. Please upload tokenizers."

    # Convert text to sequence and pad
    input_seq = kn_tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences_custom(input_seq, maxlen=max_kn_length, padding='post')

    # Predict using encoder
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = en_tokenizer.word_index.get('<start>', 1)

    stop_condition = False
    translated_sentence = ""

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = en_tokenizer.index_word.get(sampled_token_index, '')

        if sampled_word == "<end>" or len(translated_sentence.split()) > max_en_length:
            stop_condition = True
        else:
            translated_sentence += " " + sampled_word

        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]

    return translated_sentence.strip()

# Streamlit UI
st.title("Kannada to English Translator 📝")
st.write("Enter a Kannada sentence below to get the English translation.")

input_text = st.text_area("Enter Kannada text:", "", height=100)

if st.button("Translate"):
    if input_text:
        translation = translate_sentence(input_text)
        st.success(f"**Translated Text:** {translation}")
    else:
        st.warning("⚠️ Please enter a Kannada sentence.")
