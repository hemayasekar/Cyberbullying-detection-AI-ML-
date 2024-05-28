import streamlit as st
from PIL import Image

hide_menu = """
<style>
#MainMenu{
    visibility:hidden;
}

footer{
    visibility:hidden;
}
</style>
"""

showWarningOnDirectExecution = False
# image = Image.open('icons/logo.png')s)

st.markdown(hide_menu, unsafe_allow_html=True)

 
# st.sidebar.image(image , use_column_width=True, output_format='auto')
# st.sidebar.markdown("---")
# st.sidebar.markdown("<br> <br> <br> <br> <br> <br> <br> <h1 style='text-align: center; font-size: 18px; color: #0080FF;'>© 2023 | Ioannis Bakomichalis</h1>", unsafe_allow_html=True)




st.title("Support for bullying 🚨 ")
st.markdown("---")
st.markdown("<br>", unsafe_allow_html=True)

st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRbdJJjVnAyHEMx-Z6F7zlMwepsz5AN6naotrebtnEcQA&s", width=500)