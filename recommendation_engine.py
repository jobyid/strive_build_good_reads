import pandas as pd
import random

def recoomend_a_book():
    df = pd.read_csv('data/good_reads_df_web.csv')
    t = df['Title']
    title = random.choice(t)
    dft = df[df['Title']==title]

    return dft['Title'].values[0] + " by " + dft['Author'].values[0]


