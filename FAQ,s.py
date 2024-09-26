import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from datasets import load_dataset

# Load the trained BERT model and tokenizer
model_path = './bert_faq_model'
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

# Function to tokenize input text
def tokenize_text(text):
    return tokenizer(text, padding='max_length', truncation=True, return_tensors="pt")

# Define label mapping (Assumed based on your training)
label_mapping = {0: 'Label1', 1: 'Label2'}  # Update based on your unique labels

# Streamlit app UI
st.title("BERT FAQ Model")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type="csv")
if uploaded_file is not None:
    dataset = load_dataset('csv', data_files=uploaded_file)
    st.write("Dataset uploaded successfully!")

# Input for question
question = st.text_input("Enter your question:")

if st.button('Predict'):
    if question:
        # Tokenize input question
        tokenized_input = tokenize_text(question)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**tokenized_input)
            logits = outputs.logits
            predicted_label = torch.argmax(logits, dim=1).item()

        # Get the corresponding label
        predicted_class = label_mapping.get(predicted_label, "Unknown")
        
        # Display prediction
        st.write(f"Predicted Answer: {predicted_class}")
    else:
        st.write("Please enter a question.")

# Option to download the trained model
st.download_button('Download Trained Model', data=open(f"{model_path}/pytorch_model.bin", 'rb'), file_name='bert_faq_model.bin')

