from fastai.vision.all import *
import streamlit as st
import pathlib
import platform
import plotly.express as px
from PIL import Image

plt=platform.system()
if plt=='Linux': pathlib.WindowsPath = pathlib.PosixPath

st.title('Fruit Images Classification Model')
st.markdown('You can upload  or take picture of  these fruit only: Apple, pineapple, pear, grape, lemon, mango, banana and strawberry')
st.text('Do you agree?')
check=st.checkbox('I agree')

def model_result(image):
    fruit_model=load_learner('D:\\DL_model\\fruits\\venvx\\fruit_classification_model.pkl')
    predict,pred_id,probability=fruit_model.predict(image)
    st.image(image,caption=predict)
    st.success(f"Fruit: {predict}")
    st.info(f"Probability: {probability[pred_id]*100:.1f}%")    
    fig=px.bar(x=fruit_model.dls.vocab,y=probability*100) 
    st.plotly_chart(fig)

if check:
    file=st.file_uploader('Upload file here',type=['png','jpeg','gif','svg'])

    if file:
        img=PILImage.create(file)
        model_result(img)
        
    
