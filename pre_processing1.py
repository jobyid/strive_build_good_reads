import pandas as pd
import numpy as np

csv_filepath1 = './data/Strive_good_Reads_v1.csv'
csv_filepath2 = './data/goodreads_min.csv'






def pre_process1(csv_filepath1, csv_filepath2):

    df = pd.read_csv(csv_filepath1, index_col=0)
    df1 = pd.read_csv(csv_filepath2)

    df1.dropna(subset=['original_publish_year'], inplace=True)

    publish_years = df1['original_publish_year'].astype('int').values[:1000]

    title = df1['title'].apply(lambda x:x.strip())[:1000].values
    
    awards = df1['awards'].str.split(',').agg(np.size).astype('int').replace(1, np.nan).values[:1000]
   
    avg_ratings = df['minirating'].str.split('avg').apply(lambda x: x[0])
    avg_ratings =  avg_ratings.str.extract('(\d.+)')[0].apply(pd.to_numeric).apply(lambda x: int(x))[:1000].values

    num_rating = df['minirating'].str.split('avg').apply(lambda x: x[1].split(' ')[3])
    num_rating = num_rating.str.split(',').apply(lambda x: ''.join(x)).apply(pd.to_numeric)[:1000].values

    author = df['Author'][:1000].values

    #title = df['Title']

    url = df['Title_URL'][:1000].values

    data = {'url': url, 'title': title, 'author': author, 'num_ratings': num_rating, 
            'avg_rating': avg_ratings, 'awards': awards, 'original_publish_year': publish_years }

    good_read = pd.DataFrame(data)
    

    return good_read


def pre_pro2(df):
    da = df["avg_rating"]
    normalized_da =  1 + ((da - da.mean())/(da.max() - da.min())) * 9
    normalized_df_max_min = 1 + ((da - da.min())/(da.max() - da.min())) * 9
    df["norm_mean"] = normalized_da
    df["norm_max_min"] = normalized_df_max_min
    return df
    

df = pre_process1(csv_filepath1, csv_filepath2)
df2 = pre_pro2((df))
print(df2.head())

print(df2.columns)

print(df2.info())






