from flask import Flask, request, jsonify
import pytesseract
from PIL import Image
import spacy
import nltk
from firebase_admin import credentials, initialize_app, db
import logging
import requests  # To interact with Gemini API

nltk.download('punkt')     # For tokenization
nltk.download('stopwords') # For stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Initialize Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Firebase Admin SDK initialization
cred = credentials.Certificate('CREDENTIAL_json-FILE')
initialize_app(cred, {
    'databaseURL': 'YOUR-DATABASE_URL'
})

# Function to extract ingredients using SpaCy
def extract_ingredients(text):
    doc = nlp(text)
    ingredients = []
    for ent in doc.ents:
        if ent.label_ == "FOOD":
            ingredients.append(ent.text.lower())
    return ingredients

# Function to check if the product is safe using NLTK
def check_product_safety(ingredients, profile_data):
    allergens = profile_data.get('allergy', {})
    diets = profile_data.get('diet', {})
    
    # NLTK stopwords
    stop_words = set(stopwords.words('english'))
    
    for allergen, has_allergy in allergens.items():
        if has_allergy:
            for ingredient in ingredients:
                # Tokenize and filter out stop words
                tokens = word_tokenize(ingredient)
                filtered_tokens = [word.lower() for word in tokens if word.lower() not in stop_words]
                
                if allergen.lower() in filtered_tokens:
                    return f"No, this product is not safe to consume because it contains {allergen}."
    
    for diet, is_on_diet in diets.items():
        if is_on_diet:
            for ingredient in ingredients:
                # Tokenize and filter out stop words
                tokens = word_tokenize(ingredient)
                filtered_tokens = [word.lower() for word in tokens if word.lower() not in stop_words]
                
                if diet.lower() in filtered_tokens:
                    return f"No, this product is not suitable for your {diet} diet."
    
    return "Yes, this product is safe to consume."

# Function to get storage recommendations and food alternatives from Gemini API
def get_storage_and_alternatives(image_data, api_key):
    gemini_url = "YOUR_GEMINI_API_URL"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Request for storage recommendations
    storage_prompt = "Provide detailed instructions on how to store this food item to avoid health risks."
    storage_payload = {
        "instances": [{"image": image_data, "prompt": storage_prompt}]
    }
    
    response = requests.post(gemini_url, headers=headers, json=storage_payload)
    storage_recommendation = response.json().get("predictions", ["No storage information found"])[0]
    
    # Request for food alternatives
    alt_prompt = "Suggest 3 alternative healthy food items for the ingredients listed."
    alt_payload = {
        "instances": [{"image": image_data, "prompt": alt_prompt}]
    }
    
    response = requests.post(gemini_url, headers=headers, json=alt_payload)
    alternatives = response.json().get("predictions", ["No alternatives found"])[0]
    
    return storage_recommendation, alternatives

# Route to handle image uploads and process them
@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        file = request.files['file']
        user_id = request.form['user_id']  # Get user ID from the form data
        image = Image.open(file.stream)

        # Extract text from image using Tesseract
        text = pytesseract.image_to_string(image)

        # Extract ingredients using SpaCy
        ingredients = extract_ingredients(text)

        # Fetch user profile data from Firebase
        user_ref = db.reference(f'users/{user_id}')
        user_data = user_ref.get()

        if user_data:
            profile_data_ref = db.reference(f'CustomerSet/{user_data["username"]}')
            profile_data = profile_data_ref.get()
            
            if profile_data:
                # Check product safety using NLTK
                safety_message = check_product_safety(ingredients, profile_data)
                
                # Get storage recommendations and alternatives using Gemini API
                image_data = file.read()  # Convert image to binary data
                api_key = "YOUR_GEMINI_API_KEY"
                storage_recommendation, alternatives = get_storage_and_alternatives(image_data, api_key)
                
                return jsonify({
                    'ingredients': ingredients,
                    'safety_message': safety_message,
                    'storage_recommendation': storage_recommendation,
                    'alternatives': alternatives
                })
            else:
                return jsonify({'error': 'Profile data not found'}), 404
        else:
            return jsonify({'error': 'User data not found'}), 404
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

# Custom error handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Not Found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == "__main__":
    app.run(debug=True, port=8080)
