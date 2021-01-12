import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import statsmodels.api as sm

# 1. Create a 2D scatterplot with `pages` on the x-axis and `num_ratings` on the y-axis.
# 2. Can you compute numerically the correlation coefficient of these two columns?

# 9.
def plot_ratings_year():
    df = pd.read_csv('data/analyse_this.csv')
    pdf = df[['avg_rating',"original_publish_year"]]
    pdf = pdf[pdf['original_publish_year']<2020]
    pdf = pdf[pdf['original_publish_year']>200]
    pdfg = pdf.groupby(['original_publish_year']).agg(ratings =('avg_rating','mean'))
    pdfg.plot( kind = 'line', title="Ratings by year")
    plt.savefig("ratings_vs_year.png")
    plt.show()

# 10.
def awards_ratings():
    df = pd.read_csv('data/analyse_this.csv')
    pdf = df[['norm_max_min',"awards"]]
    pdf["awards"]=pdf["awards"].fillna(0)
    pdfg = pdf.groupby(["awards"]).agg(ratings =('norm_max_min','mean'))
    alt_plot_for_Awards_ratings(pdfg)
    pdfg.reset_index(inplace=True)
    pdfg.plot(kind="scatter", x="awards", y="ratings", title="Awards vs Ratings")
    print(pdfg.head())
    plt.savefig("ratings_vs_awards.png")
    plt.show()
def alt_plot_for_Awards_ratings(pdfg):
    pdfg.plot(kind='bar', title="Awards Vs Ratings")
    plt.savefig("ratings_vs_awards_alt.png")
    plt.show()

plot_ratings_year()
awards_ratings()

