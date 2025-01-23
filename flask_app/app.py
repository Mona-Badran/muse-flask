from flask import Flask, request, jsonify
import faiss
import torch
import numpy as np
import pandas as pd
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

app = Flask(__name__)

index_path = "siglip-faiss-wikiart/siglip_10k.index"
csv_path = "siglip-faiss-wikiart/wikiart_final.csv"

index = faiss.read_index(index_path)
metadata = pd.read_csv(csv_path)

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

def get_image_embedding(image):
    image = Image.open(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    embeddings = outputs[0].cpu().numpy()
    return embeddings

@app.route('/')
def home():
    return jsonify({"message": "Flask server is running successfully!"})

# Route to handle image uploads
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    query_embedding = get_image_embedding(image_file)

    query_embedding = np.expand_dims(query_embedding, axis=0)
    expected_dim = index.d
    query_embedding = np.pad(query_embedding, ((0, 0), (0, max(0, expected_dim - query_embedding.shape[1]))), mode='constant')
    
    query_embedding /= np.linalg.norm(query_embedding)
    
    k = 5
    distances, indices = index.search(query_embedding, k)
    
    result = {
        'style': 'Impressionism',
        'common_artists': ['Monet', 'Renoir'],
        'description': 'A famous art style from the 19th century.'
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)