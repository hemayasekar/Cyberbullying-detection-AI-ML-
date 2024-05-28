import streamlit as st
from PIL import Image
import base64
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
file_ = open(".\\giphy.gif", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()

st.markdown(
    f'<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><img src="data:image/gif;base64,{data_url}" alt="cat gif">',
    unsafe_allow_html=True,
)

showWarningOnDirectExecution = False
image = Image.open('.\\icons\\logo.png')

with open(".\\index.html", "r", encoding="utf-8") as f:
    html_string = f.read()

# Render HTML with unsafe_allow_html set to True
st.markdown(html_string, unsafe_allow_html=True)
st.set_page_config(page_title = "SecureNet", page_icon = image)

st.markdown(hide_menu, unsafe_allow_html=True)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.image(image , use_column_width=True, output_format='auto')


st.sidebar.markdown("---")


st.sidebar.markdown("<br> <br> <br> <br> <br> <br> <h1 style='text-align: center; font-size: 18px; color: #0080FF;'> © 2024 | Secure Net </h1>", unsafe_allow_html=True)

st.image(image , use_column_width=True)
st.image("D:\Finalyr\pra\Final-year-project-jarvis\giphy.gif")