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
                This chart shows how our the average rating of the best books ever
                 fits to the full list of the current scipy.stats distributions  
                 and also determine the distribution with the least error.
                """)
    with col2:
        st.image('fig/avg_distribution.png',use_column_width=True)
        with st.beta_expander("See explanation"):
            st.write("""
                The chart above depicts how closely distribution of average rating of books rated as best ever
                follows that of a normal distribution and that the highest point on the data distribution curve is 4.03.
                 This implies that most books had a rating around 4.03.
                """)
    with col3:
        st.image('fig/Avg_Rating_boxplot.png',use_column_width=True)
        with st.beta_expander("See explanation"):
            st.write("""
                The chart above illustrates how 50% of average ratings of books lie between 3.95 to 4.21.
                We can also see that there are a few outliers. 
                """)
    with col4:
        st.image('fig/Avg_Rating_Violin_plot.png',use_column_width=True)
        with st.beta_expander("See explanation"):
            st.write("""
                The chart above shows the relationship between average rating and if a book is 
                part of a series or not. As you can see from the figure there is not a 
                significant difference. In fact from this data we concluded that whether or not a book is in a series of books did not affect its rating.   
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


