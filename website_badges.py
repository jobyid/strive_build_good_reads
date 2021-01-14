import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import good_reads_analyse as gra
import time
import recommendation_engine as re

#git hub badge
b1, b2 = st.beta_columns([1,3])
with b1:
    st.markdown('[![](<https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github'
            '&logoColor=white>)](<https://github.com/jobyid/strive_build_good_reads>)')
#python bagde
with b2:
    st.markdown('![](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo'
                     '=python'
            '&logoColor=white)')

st.markdown('[![](<https://www.buymeacoffee.com/library/content/images/2020/09/image--67--1.png>)]('
            '<https://www.buymeacoffee.com/joby>)')

st.markdown("Need a recomendation? Use our recommeder below")
st.text_input("Enter the last book you read:")
if st.button("Reccomend"):
    st.write("The science says you should read:")
    with st.spinner("Wait For it"):
        time.sleep(2)
    st.write(re.recoomend_a_book())
