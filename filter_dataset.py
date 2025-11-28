# Tools & Assistance
#This was an individual project. I used GitHub Copilot for small code 
# assistance and asked ChatGPT and Claude some questions when I needed 
# help understanding a problem. These tools were only for support. 
# All of the experiment setup, analysis, and writing were done by me

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
import datasets
import json
import os

# already trained the hypothesis-only model & saved it in ./hypothesis_only_model/

def filter_data():
    print("Loading the hypothesis-only model...")
    model_path = "./hypothesis_only_model/"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Load raw SNLI 
    print("Loading SNLI dataset...")
    dataset = datasets.load_dataset('snli', split='train')
    
    dataset = dataset.filter(lambda ex: ex['label'] != -1)
    
    # Fn to prep data same as did for earlier 
    def prepare_hyp_only(examples):
        empty_premises = [""] * len(examples['premise'])
        tokenized = tokenizer(
            empty_premises,
            examples['hypothesis'],
            truncation=True,
            max_length=128,
            padding='max_length'
        )
        return tokenized

    trainer = Trainer(model=model)
    
    print("Getting predictions on full train set. This take some time...")
    
    tokenized_dataset = dataset.map(prepare_hyp_only, batched=True)
    preds = trainer.predict(tokenized_dataset)
    
    predicted_labels = np.argmax(preds.predictions, axis=1)
    true_labels = np.array(dataset['label'])
    
    # Check to see model guessed right using only hypothesis
    is_easy = (predicted_labels == true_labels)
    
    print(f"Total Count: {len(dataset)}")
    print(f"Easy (Artifact): {sum(is_easy)}")
    print(f"Hard (Robust): {len(dataset) - sum(is_easy)}")
    
      
    keep_indices = []
    for i in range(len(dataset)):
        if not is_easy[i]:
            keep_indices.append(i) # Always keep hard
        elif np.random.rand() < 0.5: 
            keep_indices.append(i) # Keep 50% easy
            
    filtered_dataset = dataset.select(keep_indices)
    print(f"Filtered Dataset Size: {len(filtered_dataset)}")
    
    # Save this file to use in run.py later
    output_file = "snli_filtered_train.jsonl"
    filtered_dataset.to_json(output_file)
    print(f"Saved new dataset to {output_file}")

if __name__ == "__main__":
    filter_data()