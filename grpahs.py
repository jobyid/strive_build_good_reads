import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import good_reads_analyse as gra
import time
with st.beta_container():
    st.subheader("Representing the Data")
    col1, col2, col3, col4 = st.beta_columns(4)
    with col1:
        st.image('fig/all_distributions.png',use_column_width=True)
        with st.beta_expander("See explanation"):
            st.write("""
                The chart above shows all the possible distributions we can try to fit to the data. We used this to determine the best fit. 
                """)
        st.image('fig/Best_Books_With_Highest_Rating.png',use_column_width=True)
        with st.beta_expander("See explanation"):
            st.write("")
    with col2:
        st.image('fig/avg_distribution.png',use_column_width=True)
        with st.beta_expander("See explanation"):
            st.write("""
                The chart above shows some numbers I picked for you.
                I rolled actual dice for these, so they're *guaranteed* to
                be random.
                """)
        st.image('fig/ratings_year_joint.png',use_column_width=True)
        with st.beta_expander("See explanation"):
            st.write("")
    with col3:
        st.image('fig/Avg_Rating_boxplot.png',use_column_width=True)
        with st.beta_expander("See explanation"):
            st.write("""
                The chart above shows some numbers I picked for you.
                I rolled actual dice for these, so they're *guaranteed* to
                be random.
                """)
        st.image("fig/Best_Books_With_Most_Awards.png",use_column_width=True)
        with st.beta_expander("See explanation"):
            st.write("")
    with col4:
        st.image('fig/Avg_Rating_Violin_plot.png',use_column_width=True)
        with st.beta_expander("See explanation"):
            st.write("""
                The chart above shows the relationship between average rating and if a book is 
                part of a series or not. As you can see from the figure there is not a 
                significant differance. In fact from this data we concluded that whether or not a book is in a series of books did not affect its rating.   
                """)
        st.image("fig/Best_Books_With_Highest_Reviews.png",use_column_width=True)
        with st.beta_expander("See explanation"):
            st.write("")
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


