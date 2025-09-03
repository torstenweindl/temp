import os
import streamlit as st
import datetime
import requests
from IPython.display import display
import io
from collections import Counter
import base64
import math
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

file = st.file_uploader("Upload JPG/PNG of a blood smear", type=["png","jpg","jpeg"])

if file:

    if st.button("Start analysis"):
        status_placeholder = st.empty()
        status_placeholder.text("Please stand by for about ~1m! 8 mighty CPUs are giving their best for you right now. Once processed, the results will show up below.")
        image = Image.open(file)

        original_width, original_height = image.size
        if original_height > 500:
          image_ratio = original_width / original_height
          preview_image = image.resize((int(500*image_ratio),500))
        else:
          preview_image = image
          
        
        if image.mode != 'RGB':
          image = image.convert('RGB')
        st.image(preview_image, caption="Preview")   # replaced "use_container_width=True"
        # plt.imshow(image)
        buf = io.BytesIO(); image.save(buf, format="JPEG"); buf.seek(0)
        # files_to_send = {"file": ("image.jpg", buf, "image/jpeg")}

        try:
            r = requests.post(API_URL, files={"file": ("image.jpg", buf, "image/jpeg")}, timeout=600)
            r.raise_for_status()
            data = r.json()
            # st.write("This is the test output:")

            class_mapping = {
            1: 'Basophil',
            2: 'Erythroblast',
            3: 'Monocyte',
            4: 'Myeloblast',
            5: 'Seg Neutrophil',
            6: 'Red Blood Cell'
            }

            for cell in data:
              inner_dict = data[cell]
              original_value = inner_dict['class index']
              recoded_value = class_mapping.get(original_value, 'Unknown')
              inner_dict['class index'] = recoded_value

            classes = [inner_dict['class index'] for inner_dict in data.values()]
            classes_count = Counter(classes)
            classes_count_sorted = sorted(classes_count.items(), key=lambda item: item[1], reverse=True)
            st.write(classes_count_sorted)

            total_items = len(data)
            st.write(f"#### We found {total_items} cells:")

            num_columns = 3

            data_list = list(data.items())
            num_rows = math.ceil(total_items / num_columns)

            for row in range(num_rows):
              cols = st.columns(num_columns)
              for col_index in range(num_columns):
                item_index = row * num_columns + col_index
                if item_index < total_items:
                  key, value = data_list[item_index]
                  with cols[col_index]:
                    st.write(f"##### {key}")
                    binary_data = base64.b64decode(value["image"])
                    image_stream = io.BytesIO(binary_data)
                    st.image(image_stream)
                    st.write(f"""**{value["class index"]}**""")
                    st.write(f"""(certainty: **{value["class index probability"]}**)""")
            
              
                  
            # for cell in data:
            #   st.write(f"##### {cell}")
            #   binary_data = base64.b64decode(data[cell]["image"])
            #   image_stream = io.BytesIO(binary_data)
            #   st.image(image_stream)
            #   st.write(f"""**{data[cell]["class index"]}** (certainty: **{data[cell]["class index probability"]}**)""")

          
            # st.write(data)
            # plt.imshow(data[1])
            # plt.show()

        except Exception as e:
           st.error(f"API error: {e}")
else:
    st.info("Please upload an image to start.")
