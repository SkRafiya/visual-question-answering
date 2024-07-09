import requests
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

# Load the processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

# Load the image
raw_image = Image.open(r'D:\vaq\OIP.jpg').convert('RGB')

while True:
    # Ask the user for a question
    question = input("Please enter your question about the image: ")

    # Process the inputs
    inputs = processor(raw_image, question, return_tensors="pt")

    # Generate the answer
    out = model.generate(**inputs)

    # Decode and print the answer
    print("Answer:", processor.decode(out[0], skip_special_tokens=True))

# Ask if the user wants to ask another question or exit
    next_action = input("Press 0 to exit and 1 to continue: ").strip().lower()
    if next_action != '1':
        print("Goodbye!")
        break

