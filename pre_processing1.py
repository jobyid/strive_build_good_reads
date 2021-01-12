import pandas as pd

csv_filepath = './Strive_good_Reads_v1.csv'




def pre_process1(csv_filepath):

    df = pd.read_csv(csv_filepath, index_col=0)
   
    avg_ratings = df['minirating'].str.split('avg').apply(lambda x: x[0])
    avg_ratings =  avg_ratings.str.extract('(\d.+)')[0].apply(pd.to_numeric).apply(lambda x: int(x))

    num_rating = df['minirating'].str.split('avg').apply(lambda x: x[1].split(' ')[3])
    num_rating = num_rating.str.split(',').apply(lambda x: ''.join(x)).apply(pd.to_numeric)

    author = df['Author']

    title = df['Title']

    url = df['Title_URL']

    good_read = pd.DataFrame({'url': url, 'title': title, 'author': author, 'num_ratings': num_rating, 'avg_rating': avg_ratings})
    
    #df['awards'] = df['awards'].str.split(' ').agg(np.size).astype('int')

    return good_read

def pre_pro2():
    da = df["avg_rating"]
    normalized_da =  (1 + (da - da.mean())/ (da.max() - da.min()) * 9
    normalized_df_max_min = 1 + (da - da.min())/ (da.max() - da.min()) * 9
    da["norm_mean"] = normalized_da
    da["norm_max_min"] = normalized_df_max_min
    return normalized_df
    return normalized_df_max_min


def count_awards(s):
    #takes string s and counts the "," returns the count plus 1
    return s.count(",") + 1

print(count_awards("ocsar, bafta, somthing, good book"))
