import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import good_reads_analyse as gra
import time

st.subheader("Analyse the Data")
st.markdown("The next step was to do some anaylsis of the data and create some useful tools. The "
            "best book by an author came from this step, along with producing some insights in "
            "what to visulise.")
st.markdown("**Here's the code for the best book by Author tool**")
st.code('''df = pd.read_csv('data/analyse_this.csv')

def my_best_book(author):
    if df['author'].str.contains(author).sum() > 0:
        autors_books = df[df['author']== author]
        rating = autors_books.loc[autors_books['norm_max_min'].idxmax()]
        return rating.title
    return "Author not found"''')
with st.beta_expander("See explanation"):
            st.write(""" The code is actually pretty simple, using the power of pandas we can 
            take the main data frame and author name input, then create a smaller dataframe with 
            only the books written by that author. Then it is just the case of finding the one 
            with the highest rating (usng min max norm rating)""")
st.write("")
st.write("")
st.markdown("**If you would like to you can try the tool out in the sidebar**")
