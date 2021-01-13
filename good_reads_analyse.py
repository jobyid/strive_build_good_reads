import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('data/analyse_this.csv')

def my_best_book(author):
    if df['author'].str.contains(author).sum() > 0:
        autors_books = df[df['author']== author]
        rating = autors_books.loc[autors_books['norm_max_min'].idxmax()]
        return rating.title
    return "Author not found"

def awards(df):
    data=df
    df = data.groupby("awards")["awards"].count()
    print(df)


def original_publish_year(df):
    data=df
    dfs = data.groupby("original_publish_year")["num_ratings"].mean()
    print(dfs)
#original_publish_year(df)
