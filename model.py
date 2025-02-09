from transformers import AutoModelForSeq2SeqLM

# Load pretrained BLIP model
def load_model():
    model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/blip-image-captioning-large")
    return model

