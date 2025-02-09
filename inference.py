from PIL import Image
from transformers import pipeline

# Load trained model
generator = pipeline("image-to-text", model="image-to-html-model")

# Function to generate HTML from an image
def generate_html(image_path):
    image = Image.open(image_path)
    html_output = generator(image)
    return html_output

# Example usage
if __name__ == "__main__":
    image_path = "sample.jpg"  # Change to your test image
    generated_html = generate_html(image_path)
    print("Generated HTML:", generated_html)

