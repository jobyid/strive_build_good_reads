import pandas as pd
import numpy as np




def pre_process1(csv_filepath1, csv_filepath2, csv_filepath3, csv_filepath4):

    df = pd.read_csv(csv_filepath1, index_col=0)
    df1 = pd.read_csv(csv_filepath2)
    df2 = pd.read_csv(csv_filepath3)
    df3 = pd.read_csv(csv_filepath4)

    df1.dropna(subset=['original_publish_year'], inplace=True)

    publish_years = df1['original_publish_year'].astype('int').values[:900]

    title = df1['title'].apply(lambda x:x.strip())[:900].values
    
    awards = df1['awards'].str.split(',').agg(np.size).astype('int').replace(1, np.nan).values[:900]
   
    avg_ratings = df['minirating'].str.split('avg').apply(lambda x: x[0])
    avg_ratings =  avg_ratings.str.extract('(\d.+)')[0].apply(pd.to_numeric).apply(lambda x: int(x))[:900].values

    num_rating = df['minirating'].str.split('avg').apply(lambda x: x[1].split(' ')[3])
    num_rating = num_rating.str.split(',').apply(lambda x: ''.join(x)).apply(pd.to_numeric)[:900].values

    author = df['Author'][:900].values

    genres = df3['genre']

    locations = df3['locations']

    num_pages = df3['num_pages']

    num_reviews = df2['num_reviews'][:900]
    #num_rating = df['num_ratings'][:900]
    is_series = df2['series'][:900]

    #title = df['Title']

    url = df['Title_URL'][:900].values

    data = {'url': url, 'title': title, 'author': author, 'num_ratings': num_rating, 
            'avg_rating': avg_ratings, 'awards': awards, 'original_publish_year': publish_years,
            'num_reviews': num_reviews, 'is_series': is_series, 'genre': genres, 'location': locations, 'num_pages': num_pages,}

    good_read = pd.DataFrame(data)
    

    return good_read


def pre_pro2(df):
    da = df["avg_rating"]
    normalized_da =  1 + ((da - da.mean())/(da.max() - da.min())) * 9
    normalized_df_max_min = 1 + ((da - da.min())/(da.max() - da.min())) * 9
    df["norm_mean"] = normalized_da
    df["norm_max_min"] = normalized_df_max_min
    return df
    

csv_filepath1 = './data/Strive_good_Reads_v1.csv'
csv_filepath2 = './data/goodreads_min.csv'
csv_filepath3 = 'data/review_ratings_series.csv'
csv_filepath4 = 'data/pages_genre_loc.csv'

df = pre_process1(csv_filepath1, csv_filepath2, csv_filepath3, csv_filepath4)
df2 = pre_pro2((df))
#print(df2.head())

#print(df2.columns)

#print(df2.info())

df2.to_csv('./data/analyse_this.csv', index=False)






