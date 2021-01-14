import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
df = pd.read_csv('data/analyse_this.csv')


st.title("Explore our collection!")

st.header("Reading. Data Science supported.")

st.subheader("Are you looking for a really good read that is worth your while? Think no more, choose one of the following from our cross checked database.")



st.write("Top 1000 books recommended by top critics")
st.dataframe(df)


# table
st.sidebar.title("1000 Good reads")
st.sidebar.write("More time reading. Less time searching")

st.sidebar.image("fig/5231.jpg",use_column_width = True)


st.write("In 1000 good reads we believe that reading should not be a complicated, stressful or even unsatisfying hobbie."
         " If you are not certain on where to put your money, you can relax and trust on our Data Science team."
         "We are there to pre-select the best reads")


st.image("fig/12.jpg", use_column_width = True)


with st.beta_container():
    st.subheader("Representing the Data")
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
                """)
    with col3:
        st.image('fig/Avg_Rating_boxplot.png',use_column_width=True)
        with st.beta_expander("See explanation"):
            st.write("""
                The chart above shows some numbers I picked for you.
                I rolled actual dice for these, so they're *guaranteed* to
                be random.
                """)
    with col4:
        st.image('fig/Avg_Rating_Violin_plot.png',use_column_width=True)
        with st.beta_expander("See explanation"):
            st.write("""
                The chart above shows the relationship between average rating and if a book is 
                part of a series or not. As you can see from the figure there is not a 
                significant differance. In fact from this data we concluded that whether or not a book is in a series of books did not affect its rating.   
                """)

with st.beta_container():
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
#Graph

#Our collection contains an average rating of 4 to 4.2 stars with a certainty of 70% . So rest assured you'll be getting a top rated book-feed without taking the effort of researching.

#Graphs awards-ratings

#Despite the awards count, you can rely on us to select good quality books for your mind.


#use places column into a button that shows place of the book

#Hypothesis = awards have increased year by year - we are aware of this

#compare awards from the same year. If they are from different year : compare each one to the mean of the year

#also with genre



