from fastai.vision.all import *
import streamlit as st
import platform
import plotly.express as px

import pathlib
plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

    
## title
st.title('Jihozlarini klassifikatsiya qiluvchi model')

st.text('Ushbu model ishxona, musiqa va oshxona jihozlarini klassifikatsiya qiladi')

# rasmni joylash
file = st.file_uploader('Rasm yuklash', type=['png','jpeg','gif','svg','jpg'])
if file:
    with st.spinner(text='In progress'):
        time.sleep(1)
    st.image(file)
    # PIL konvert
    img = PILImage.create(file)
    # model
    model = load_learner('new_model.pkl')

    #predict
    pred, prod_id , probs  = model.predict(img)
    st.success(f"Bashorat: {pred}")
    st.info(f"Probability: {probs[prod_id]*100:0.1f}%")

    #plotting
    fig = px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)


