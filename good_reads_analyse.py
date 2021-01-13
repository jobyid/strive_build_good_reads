import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('data/analyse_this.csv')
# question number 7
def my_best_book(author):
    autors_books = df[df['author']== author]
    rating = autors_books.loc[autors_books['norm_max_min'].idxmax()]
    return rating.title

my_best_book('Jane Austen')

def awards(df):
    data=df
    df = data.groupby("awards")["awards"].count()
    print(df)
    
    
def original_publish_year(df):
    data=df
    df = data.groupby("original_publish_year")["num_ratings"].mean()
    print(df)
