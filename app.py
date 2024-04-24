from flask import Flask, render_template, request, send_file
import cv2
import numpy as np
from io import BytesIO

app = Flask(__name__)

# Function to cartoonize an image using grayscale
def cartoonize_grayscale(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filtering to reduce noise while preserving edges
    cartoon = cv2.bilateralFilter(gray, 9, 75, 75)

    return cartoon

# Function to cartoonize an image using edge detection
def cartoonize_edge_detection(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect edges using Canny edge detection
    edges = cv2.Canny(gray, 100, 200)

    return edges

# Function to cartoonize an image
def cartoonize(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filtering to reduce noise while preserving edges
    cartoon = cv2.bilateralFilter(gray, 9, 75, 75)

    # Detect edges using adaptive thresholding
    edges = cv2.adaptiveThreshold(cartoon, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 10)

    # Apply color quantization to reduce the number of colors in the image
    num_colors = 16
    Z = img.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    ret, label, center = cv2.kmeans(Z, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    quantized_img = center[label.flatten()]
    quantized_img = quantized_img.reshape(img.shape)

    # Combine the edges with the quantized image to create the cartoon effect
    cartoon = cv2.bitwise_and(quantized_img, quantized_img, mask=edges)


    return cartoon

# Route to render the upload form
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle file upload and cartoonization
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part', 400

    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    model = request.form['model']

    if file:
        # Read the uploaded image
        img_stream = file.stream
        img_array = np.frombuffer(img_stream.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Cartoonize the image based on the selected model
        if model == 'grayscale':
            cartoon_image = cartoonize_grayscale(img)
        elif model == 'edge_detection':
            cartoon_image = cartoonize_edge_detection(img)
        elif model == 'cartoonize':
            cartoon_image = cartoonize(img)
        else:
            return 'Invalid model selected', 400

        # Convert the cartoon image to bytes
        _, img_encoded = cv2.imencode('.jpg', cartoon_image)
        img_bytes = BytesIO(img_encoded)

        # Return the cartoonized image
        return send_file(img_bytes, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
