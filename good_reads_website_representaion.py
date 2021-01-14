import good_reads_analyse as gra
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
df = pd.read_csv('data/analyse_this.csv')


st.title("Explore our collection!")

st.header("Reading. Data Science supported.")
st.write(" ")
st.subheader("Are you looking for a really good read that is worth your while? Think no more, choose one of the following from our cross checked database.")
st.write(" ")
st.image("fig/12.jpg", use_column_width = True)
st.write(" ")

st.subheader("Top 1000 books recommended by top readers")
st.dataframe(df)
st.write(" ")

# table
st.sidebar.title("1000 Good reads")
st.sidebar.write("More time reading. Less time searching")

st.sidebar.image("fig/5231.jpg",use_column_width = True)

st.write(" ")
st.write("In 1000 Good Reads we believe that reading should not be a complicated, stressful or even an unsatisfying hobbie."
         " If you are not certain on where to put your money, you can **relax** and trust on our Data Science team."
         " We are there to pre-select the **best reads**")
st.write(" ")



with st.beta_container():
    st.subheader("How do we select the BEST of the best?")
    col1, col2, col3, col4 = st.beta_columns(4)
    with col1:
        st.image('fig/all_distributions.png',use_column_width=True)
        with st.beta_expander("See explanation"):
            st.write("""
                The chart above shows some numbers I picked for you.
                I rolled actual dice for these, so they're *guaranteed* to
                be random.
                """)
    with col2:
        st.image('fig/avg_distribution.png',use_column_width=True)
        with st.beta_expander("See explanation"):
            st.write("""
                The chart above shows some numbers I picked for you.
                I rolled actual dice for these, so they're *guaranteed* to
                be random.
                Gain Access to **Top Quality** content. Our books are placed within the **95% best titles** of all time!
                """)
    with col3:
        st.image('fig/Avg_Rating_boxplot.png',use_column_width=True)
        with st.beta_expander("See explanation"):
            st.write("""
                The chart above shows some numbers I picked for you.
                I rolled actual dice for these, so they're *guaranteed* to
                be random.
                Our collection is based on an average **4.1/5** rating points with a **70% chance**.
                """)
    with col4:
        st.image('fig/Avg_Rating_Violin_plot.png',use_column_width=True)
        with st.beta_expander("See explanation"):
            st.write("""
                The chart above shows the relationship between average rating and if a book is 
                part of a series or not. As you can see from the figure there is not a 
                significant differance. In fact from this data we concluded that whether or not a book is in a series of books did not affect its rating.   
                """)

with st.sidebar.beta_container():
    st.subheader("What's an Authors Best Rated Book? Find Out Below")
    auth1, auth2 = st.beta_columns([3, 1])
    with auth1:
        title = st.text_input('Enter an Author Name to find their best rated book')
    with auth2:
        st.write("")
        st.write("")
        search = st.button("Search")
    if search:
        st.write(title, "'s best rated book is: ",gra.my_best_book(title))
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")

st.write("So,what are you waiting for?   **Subscribe Now!**   and get a **30% discount** on your first month. ")

#Our collection contains an average rating of 4 to 4.2 stars with a certainty of 70% . So rest assured you'll be getting a top rated book-feed without taking the effort of researching.

#Graphs awards-ratings

#Despite the awards count, you can rely on us to select good quality books for your mind.





