

import streamlit as st
import pandas as pd
from io import StringIO
from loadmodel import Model_generate
import streamlit.components.v1 as components
import numpy as np
from PIL import Image
import os

model_name = os.getenv("MODEL_NAME")
device = os.getenv("DEVICE")
model = Model_generate(model_name=model_name, device=device)








html_temp = """
<div style = "background.color:teal; padding:10px">
<h2 style = "color:white; text_align:center;"> Saturn Cloud Demo - Text to image</h2>
<p style = "color:white; text_align:center;"> </p>
</div>
"""
st.markdown(html_temp, unsafe_allow_html = True)


#st.cache()
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """

text = st.text_input("Input your prompt")


import time
import streamlit as st

with st.spinner('Wait for it...'):
    

    if st.button('Generate Image'):
        if len(text) == 0:
            st.write("input your prompt")
        else:
            pass
        result = model.generate_image(prompt=text)
        image = Image.open("generatedimages/"+str(text)+'.jpeg')
        st.image(image, caption=text)


    st.success('Done!')
