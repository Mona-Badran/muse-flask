from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"message": "Flask server is running successfully!"})

# Route to handle image uploads
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']
    
    result = {
        'style': 'Impressionism',
        'common_artists': ['Monet', 'Renoir'],
        'description': 'A famous art style from the 19th century.'
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)