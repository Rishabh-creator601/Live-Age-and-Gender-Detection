
import streamlit as st 

st.markdown("<h1 style='text-align: center; color:Blue;'>Model Analysis </h1>", unsafe_allow_html=True)

    
    
st.header("Model Details ")
st.markdown("- Model has Accuracy of ~87%")
st.markdown("- Model is trained with a simple shallow convolutional network")
st.download_button(
    label="Download Gender Model",
    data="./models/gender_model.hdf5",
    file_name="gender_model.hdf5",
)

image1 =  "./images/gender_model.png"
image2  = "./images/gender_metrics.png"


col1,col2  = st.columns(2)


with col1:
    st.header("Model Architecture")
    st.image(image1)

with col2:
    st.header("Model Progress")
    st.image(image2)
    

    
    
