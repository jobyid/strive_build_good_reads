import good_reads_analyse as gra
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
df = pd.read_csv('data/good_reads_df_web.csv',index_col=0)
da = pd.read_csv('data/awards_by_year.csv')



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

st.title("This is how we work")
st.subheader("Scraping the Data")
st.markdown("The data for our exploration of the best reads came from the website [Good Reads]("
            "<https://www.goodreads.com>), unfortunately the API for the website has been "
            "discontinued, so we scrapped the data. To do this we used [Octoparse]("
            "<https://octoparse.com>). Each of the Octoparse scripts we used are [here]("
            "<https://github.com/jobyid/strive_build_good_reads/tree/main/Scrape>) one of the "
            "exciting things about using Octoparse was the bility to clean the data as the "
            "scraping process happened. In the end we were left with a string of csv files")
st.subheader("Processing the Data")
st.markdown("With our freshly created set of CSV files we need to work some pandas magic and "
            "wrangle this into something useful. Cleaning data was key, although much of this had been done in scraping, then we needed to make some dataframes for analysis. We wanted to explore how the ratings played a role im the rankings, so we performed some normalisations on the ratings data.")
st.markdown("The code we used fort his process can be found [here](<https://github.com/jobyid/strive_build_good_reads/blob/main/good_reads_preprocessing.py>) ")
st.markdown("**Here a selection of functions  you might like**")
code1, code2 = st.beta_columns(2)
with code1:
    st.code('''def mean_minmax_normalisation(df):'
            'da = df["avg_rating"]'
            'normalized_da =  1 + ((da - da.mean())/(da.max() - da.min())) * 9'
            'normalized_df_max_min = 1 + ((da - da.min())/(da.max() - da.min())) * 9'
            'df["norm_mean"] = normalized_da'
            'df["norm_max_min"] = normalized_df_max_min'
            'return df''')
with code2:
    st.code('''def pre_process(csv_filepath1, csv_filepath2, csv_filepath3, csv_filepath4):

    df = pd.read_csv(csv_filepath1, index_col=0)
    df1 = pd.read_csv(csv_filepath2)
    df2 = pd.read_csv(csv_filepath3)
    df3 = pd.read_csv(csv_filepath4)

    df1.dropna(subset=['original_publish_year'], inplace=True)

    publish_years = df1['original_publish_year'].astype('int').values[:900]

    title = df1['title'].apply(lambda x:x.strip())[:900].values

    awards = df1['awards'].str.split(',').agg(np.size).astype('int').replace(1, np.nan).values[:900]

    avg_ratings = df['minirating'].str.split('avg').apply(lambda x: x[0])
    avg_ratings =  avg_ratings.str.extract('(\d.+)')[0].apply(pd.to_numeric)[:900].values

    num_rating = df['minirating'].str.split('avg').apply(lambda x: x[1].split(' ')[3])
    num_rating = num_rating.str.split(',').apply(lambda x: ''.join(x)).apply(pd.to_numeric)[:900].values

    author = df['Author'][:900].values

    genres = df3['genre']

    locations = df3['locations']

    num_pages = df3['num_pages']

    num_reviews = df2['num_reviews'][:900]
    #num_rating_f = df['num_rating'][:900]
    is_series = df2['series'][:900]

    #title = df['Title']

    url = df['Title_URL'][:900].values

    data = {'url': url, 'title': title, 'author': author, 'num_ratings': num_rating,
            'avg_rating': avg_ratings, 'awards': awards, 'original_publish_year': publish_years,
            'num_reviews': num_reviews, 'is_series': is_series, 'genre': genres, 'location': locations, 'num_pages': num_pages}

    good_read = pd.DataFrame(data)


    return good_read''')
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

st.write("")
st.write("")
st.write("")
st.write("")
st.write("")
st.dataframe(da)


