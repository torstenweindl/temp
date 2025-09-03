import os
import streamlit as st
import datetime
import requests
import io
import pandas as pd
from google.cloud import storage
from params import *
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# API_URL = st.secrets["API_URL"]    # API_URL stored in a local "secrets" file; in production, API_URL will be stored in Streamlit's secrets section in the web interface
# API_URL = 'http://127.0.0.1:8000/segment/'  # API URL hardcoded to the local server for the time being
API_URL = st.secrets['API_URL']
BUCKET_NAME = st.secrets['BUCKET_NAME']
# GCP_PROJECT = st.secrets['GCP_PROJECT']


model_list = ['20250826-152119.keras','20250827-141827.keras','20250828-082537.keras','20250828-083337.keras','20250828-083558.keras','20250828-154636_model BW.keras','20250829-130807_model BW.keras',
              '20250902-193012_model BW.keras','20250902-193056_ModelTrainedOnSegData_simpleCNN_v1.keras','20250903-073842_ModelTrainedOnSegData_simpleCNN_v1.keras','20250903-091757_ModelTrainedOnSegData_simpleCNN_final.keras']

# client = storage.Client()
# bucket = client.bucket(BUCKET_NAME)
# blobs_list = list(bucket.list_blobs(prefix="models/"))
# sorted_blobs = sorted(blobs_list, key=lambda x: x.updated, reverse=True)
# list_of_blobs = [blob.name for blob in sorted_blobs]
# selected_model = st.selectbox("**Choose the model to be used**" + "\n\n" + "*(list sorted descending by model deployment date - latest model is preselected)*", list_of_blobs)
# data_to_send = {"option": selected_model}

# data_to_send = {"option_1": "no value selected", "option_2": "no value selected", "option_3": "no value selected"}

file = st.file_uploader("Upload JPG/PNG - and this is just a random text", type=["png","jpg","jpeg"])

if file:

    if st.button("Segment"):
        status_placeholder = st.empty()
        status_placeholder.text("Segmenting in the works ... please stand by.")
        image = Image.open(file)
        buf = io.BytesIO(); image.save(buf, format="JPEG"); buf.seek(0)
        files_to_send = {"file": ("image.jpg", buf, "image/jpeg")}

        try:
            r = requests.post(API_URL, files=files_to_send, timeout=600)
            r.raise_for_status()
            data = r.json()
            st.write("This is the test output:")
            st.write(data)
            # plt.imshow(data[1])
            # plt.show()

        except Exception as e:
           st.error(f"API error: {e}")
else:
    st.info("Please upload an image to start.")
