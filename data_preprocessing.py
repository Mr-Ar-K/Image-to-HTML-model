from datasets import load_dataset
from transformers import AutoProcessor

# Load dataset
dataset = load_dataset("HuggingFaceM4/WebSight", "v0.2")

# Initialize tokenizer
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")

# Tokenize HTML function
def preprocess_data(sample):
    encoding = processor.tokenizer(sample["html"], padding="max_length", truncation=True, return_tensors="pt")
    return {"pixel_values": sample["image"], "labels": encoding["input_ids"].squeeze()}

# Apply preprocessing
train_data = dataset["train"].map(preprocess_data)

