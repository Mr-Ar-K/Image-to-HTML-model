from nltk.translate.bleu_score import sentence_bleu
import torch

# Load trained model
from model import load_model
model = load_model()

# Function to evaluate BLEU score
def evaluate_bleu(predicted_html, actual_html):
    reference = [actual_html.split()]
    candidate = predicted_html.split()
    score = sentence_bleu(reference, candidate)
    return score

# Example usage
pred_html = "<html><body><h1>Title</h1></body></html>"
actual_html = "<html><body><h1>Title</h1></body></html>"
print("BLEU Score:", evaluate_bleu(pred_html, actual_html))


