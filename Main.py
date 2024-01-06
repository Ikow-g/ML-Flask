import base64
import io  # Import the 'io' module
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model('C:/Machine Learning/Machine Learning/modeltes.h5')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        image_data = request.json.get('image')
        # image_path = 'C:/Users/User/Downloads/Machine Learning/Machine Learning/putri.jpg'

        # # Open the image file in binary mode
        # with open(image_path, 'rb') as image_file:
        #     # Read the binary data of the image
        #     image_binary_data = image_file.read()

        # # Encode the binary data as base64
        # base64_encoded = base64.b64encode(image_binary_data).decode('utf-8')

        # # Directly use the base64-encoded string
        # image_data = base64_encoded

        # # Decode base64 image data

        decoded_image_data = base64.b64decode(image_data)

        # Use the decoded binary data directly for image processing
        img = image.load_img(io.BytesIO(decoded_image_data), target_size=(200, 200))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize the image

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)

        # Format response
        response = {'prediction': int(predicted_class)}

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000, debug=True)