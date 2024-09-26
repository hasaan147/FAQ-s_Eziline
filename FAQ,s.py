import streamlit as st
from datasets import Dataset
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch

# Load the pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Function to tokenize the input text
def tokenize_function(examples):
    return tokenizer(examples['Question'], padding='max_length', truncation=True)

# Function to map labels
def map_labels(example, label_mapping):
    key = tuple(example['Answer'])
    if key not in label_mapping:
        key = tuple([example['Answer'][0]])  # Handle case where key isn't found
    example['label'] = label_mapping[key]
    return example

# Main Streamlit app function
def main():
    st.title("BERT Fine-tuning with Custom Dataset")

    # File upload option
    uploaded_file = st.file_uploader("Upload CSV Dataset", type="csv")
    
    if uploaded_file is not None:
        # Read the uploaded CSV file into a pandas DataFrame
        df = pd.read_csv(uploaded_file)
        
        # Convert the pandas DataFrame to a Hugging Face Dataset
        dataset = Dataset.from_pandas(df)

        # Tokenize dataset
        st.write("Tokenizing the dataset...")
        tokenized_datasets = dataset.map(tokenize_function, batched=True)

        # Mapping labels
        st.write("Mapping labels...")
        unique_labels = list(set([tuple(answer) for answer in tokenized_datasets['Answer']]))
        label_mapping = {}
        for idx, label in enumerate(unique_labels):
            label_mapping[label] = idx
            if len(label) > 1:
                for sub_label in label:
                    label_mapping[tuple([sub_label])] = idx

        tokenized_datasets = tokenized_datasets.map(lambda x: map_labels(x, label_mapping), batched=False)

        # Train/Test split
        st.write("Splitting dataset into train/test sets...")
        tokenized_datasets = tokenized_datasets.train_test_split(test_size=0.2)

        # Load the pre-trained BERT model for sequence classification
        st.write("Loading BERT model...")
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(unique_labels))

        # Training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01
        )

        # Define the Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['test'],
            tokenizer=tokenizer,
            compute_metrics=None  # Add metrics if needed
        )

        # Train the model
        st.write("Training the model...")
        trainer.train()

        st.write("Training complete!")

if __name__ == "__main__":
    main()
