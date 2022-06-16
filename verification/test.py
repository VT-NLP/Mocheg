from PIL import Image
import requests
import numpy as np
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer



def test1():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    image1 = Image.open("data/test/images/00000-00000-00-8274137963_fb7690b164_b.jpg")
    image2 = Image.open("data/test/images/00000-proof-03-hotsprings.jpg")
    # image3 = Image.open("3.jpg")
    # image4 = Image.open("2.jpg")

    image = [image1, image2]# image3, image4]

    text1 = 'Hello'
    text2 = 'Im good, how about u'

    text = [text1, text2]

    inputs = processor(text = text, images=image, return_tensors="pt", padding=True)

    print(inputs.pixel_values.shape)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
    
    
    
test1()