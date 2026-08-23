import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Handwritten Digit Recognizer",
    page_icon="🔢",
    layout="centered"
)

# Load trained model
model = tf.keras.models.load_model("digit_model.keras")

# App title
st.title("🔢 Handwritten Digit Recognizer")

st.write(
    "Upload an image of a handwritten digit (0–9), "
    "and the CNN will predict it."
)

st.success("Model loaded successfully! ✅")

# Upload image
uploaded_file = st.file_uploader(
    "Upload a handwritten digit image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    # Display uploaded image
    st.image(
        uploaded_file,
        caption="Uploaded Image",
        width=200
    )

    # Open image and convert to grayscale
    image = Image.open(uploaded_file).convert("L")

    # Convert image to NumPy array
    original = np.array(image)

    # Find dark pixels
    threshold = np.percentile(original, 35)

    # Convert dark digit to white
    # Convert light background to black
    binary = np.where(
        original < threshold,
        255,
        0
    ).astype(np.uint8)

    # Convert back to PIL image
    image = Image.fromarray(binary)

    # Find digit area
    coords = np.argwhere(binary > 0)

    if coords.size > 0:

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        # Crop around digit
        cropped = image.crop(
            (
                x_min,
                y_min,
                x_max + 1,
                y_max + 1
            )
        )

        # Make image square
        width, height = cropped.size
        size = max(width, height)

        square = Image.new(
            "L",
            (size, size),
            0
        )

        paste_x = (size - width) // 2
        paste_y = (size - height) // 2

        square.paste(
            cropped,
            (paste_x, paste_y)
        )

        # Resize to MNIST size
        image = square.resize((28, 28))

    # Normalize pixel values
    image_array = np.array(image) / 255.0

    # Add batch and channel dimensions
    image_array = image_array.reshape(
        1,
        28,
        28,
        1
    )

    # Make prediction
    predictions = model.predict(
        image_array,
        verbose=0
    )

    # Get predicted digit
    predicted_digit = np.argmax(predictions[0])

    # Get confidence
    confidence = np.max(predictions[0]) * 100

    # Display prediction
    st.subheader(
        f"🎯 Predicted Digit: {predicted_digit}"
    )

    st.write(
        f"Confidence: **{confidence:.2f}%**"
    )