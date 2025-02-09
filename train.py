import torch
from transformers import TrainingArguments, Trainer
from data_preprocessing import train_data
from model import load_model

# Load model
model = load_model()

# Training parameters
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    num_train_epochs=5
)

# Train model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data
)

trainer.train()

# Save trained model
model.save_pretrained("image-to-html-model")

