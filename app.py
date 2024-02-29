

import streamlit as st
import tensorflow_hub as hub
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

from PIL import Image
import numpy as np
st.title("Sea Animals Image Classification")

st.write("Predict the Sea Animal that is being represented in the image.")

model = load_model("seaImagemodel.h5",custom_objects={'KerasLayer':hub.KerasLayer})
labels = {
      0: 'Clams',
    1: 'Corals',
    2: 'Crabs',
    3: 'Dolphin',
    4: 'Eel',
    5: 'Fish',
    6: 'Jelly Fish',
    7: 'Lobster',
    8: 'Nudibranches',
    9: 'Octopus',
    10: 'Otter',
    11: 'Penguin',
    12: 'Puffers',
    13: 'Sea Rays',
    14: 'Sea Urchins',
    15: 'Seahorse',
    16: 'Seal',
    17: 'Sharks',
    18: 'Shrimp',
    19: 'Squid',
    20: 'Starfish',
    21: 'Turtle_Tortoise',
    22: 'Whale'
}
uploaded_file = st.file_uploader(
    "Upload an image of a Sea Animal:", type=['jpg','png','jpeg']
)
predictions=-1
if uploaded_file is not None:
    image1 = Image.open(uploaded_file)
    image1=image.smart_resize(image1,(224,224))
    img_array = image.img_to_array(image1)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array/255.0
    predictions = model.predict(img_array)
    label=labels[np.argmax(predictions)]


st.write("### Prediction Result")
if st.button("Predict"):
    if uploaded_file is not None:
        image1 = Image.open(uploaded_file)
        st.image(image1, caption="Uploaded Image", use_column_width=True)
        st.markdown(
            f"<h2 style='text-align: center;'>Image of {label}</h2>",
            unsafe_allow_html=True,
        )
    else:
        st.write("Please upload file or choose sample image.")


