from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Load the trained model
model = load_model('bird_species_new_model1.h5')

# Define a function to preprocess the image (resize to 150x150)
def prepare_image(img_path):
    img = Image.open(img_path).convert('RGB')  # Ensure the image is in RGB mode
    img = img.resize((150, 150))  # Resize to the model's expected input size (150x150)
    img = np.array(img) / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Species and their descriptions
species_descriptions = {
    "AMERICAN GOLDFINCH": "The American Goldfinch, known for its vibrant yellow feathers and cheerful song, is a favorite among backyard birdwatchers. These small songbirds are unique for their late breeding season, often aligning with the abundance of thistle plants. Their diet consists almost entirely of seeds, making them vegetarians of the bird world.",
    "BARN OWL": "The Barn Owl, with its heart-shaped face and ghostly appearance, is a master of the night skies. Known for its silent flight and exceptional hunting skills, it preys on rodents using its acute sense of hearing. Found worldwide, the Barn Owl is an icon of mystery and beauty in folklore.",
    "CARMINE BEE-EATER": "The Carmine Bee-Eater is a striking bird with vivid red plumage and turquoise underparts, often seen perched on branches near rivers. Native to Africa, this bird lives up to its name by skillfully catching bees and other insects in mid-air. Its sociable nature and large nesting colonies make it a fascinating sight.",
    "DOWNY WOODPECKER": "The Downy Woodpecker is the smallest woodpecker in North America but packs a punch with its rapid drumming. Recognized by its black-and-white pattern and red cap, this bird thrives in forests, backyards, and even city parks, where it feeds on insects and seeds.",
    "EMPEROR PENGUIN": "The Emperor Penguin is the tallest and heaviest of all penguin species, renowned for surviving the harsh Antarctic winters. These majestic birds huddle together in large groups to stay warm, showcasing teamwork and resilience. Their incredible diving ability allows them to hunt fish at depths of up to 500 meters.",
    "FLAMINGO": "Flamingos are iconic for their long legs, S-shaped necks, and vibrant pink feathers, which they get from the carotenoids in their diet. Found in large flocks in saline or alkaline lakes, flamingos are highly social and perform intricate group dances during mating rituals. Their unique feeding style involves filtering water upside-down with their specialized beaks"
}

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    # Save the uploaded file
    uploads_dir = './static/uploads'
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)
        
    img_path = os.path.join(uploads_dir, file.filename)
    file.save(img_path)

    # Prepare the image (resize it to 150x150)
    img = prepare_image(img_path)

    # Make prediction
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)  # Get the index of the predicted class

    # Replace with actual species mapping if you have a list of species
    species = ["AMERICAN GOLDFINCH", "BARN OWL", "CARMINE BEE-EATER", "DOWNY WOODPECKER", "EMPEROR PENGUIN", "FLAMINGO"]  # Example species list
    predicted_species = species[predicted_class]
    
    # Get the description of the predicted species
    description = species_descriptions.get(predicted_species, "Description not available.")
    
    return render_template('index.html', predicted_species=predicted_species, description=description, img_path=img_path)

if __name__ == '__main__':
    app.run(debug=True)
