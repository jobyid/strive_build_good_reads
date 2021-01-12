import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st

# 1. Create a 2D scatterplot with `pages` on the x-axis and `num_ratings` on the y-axis.
# 2. Can you compute numerically the correlation coefficient of these two columns?

# 9.
def plot_ratings_year():
    df = pd.read_csv('data/analyse_this.csv')
    pdf = df[['norm_mean',"original_publish_year"]]
    pdf = pdf[pdf['original_publish_year']<2020]
    pdf = pdf.sort_values(by='original_publish_year')
    pdf.plot(x='norm_mean',y="original_publish_year", kind = 'scatter')
    plt.show()

# 10.
def awards_ratings():
    df = pd.read_csv('data/analyse_this.csv')
    pdf = df[['norm_max_min',"awards"]]
    pdf["awards"]=pdf["awards"].fillna(0)
    pdf.plot(x="awards",y="norm_max_min",kind="scatter")
    plt.show()


plot_ratings_year()
awards_ratings()

