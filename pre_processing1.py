import pandas as pd

csv_filepath = './Strive_good_Reads_v1.csv'




def pre_process1(csv_filepath):

    df = pd.read_csv(csv_filepath, index_col=0)
   
    avg_ratings = df['minirating'].str.split('avg').apply(lambda x: x[0])
    num_rating = df['minirating'].str.split('avg').apply(lambda x: x[1].split(' ')[3])
    author = df['Author']
    title = df['Title']
    url = df['Title_URL']
    good_read = pd.DataFrame({'url': url, 'title': title, 'author': author, 'num_ratings': num_rating, 'avg_rating': avg_ratings})
    
    #df['awards'] = df['awards'].str.split(' ').agg(np.size).astype('int')

    return good_read

