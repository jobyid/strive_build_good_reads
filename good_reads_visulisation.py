import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import statsmodels.api as sm
import seaborn as sns
df = pd.read_csv('data/analyse_this.csv')


# Question 9

def ratings_per_year_joint_plot():
    pdf = df[['avg_rating',"original_publish_year"]]
    pdf = pdf[pdf['original_publish_year']<2022]
    pdf = pdf[pdf['original_publish_year']>200]
    sns.set_style('darkgrid')
    sns.set(rc={'figure.figsize':(9,6)})
    sns.jointplot(x='original_publish_year', y='avg_rating', data=pdf)
    plt.show()


# Question 10
def awards_ratings():
    pdf = df[['norm_max_min',"awards"]]
    pdf["awards"]=pdf["awards"].fillna(0)
    pdf = pdf[pdf['awards']>0]
    pdfg = pdf.groupby(["awards"]).agg(ratings =('norm_max_min','mean'))
    pdfg.reset_index(inplace=True)
    df.plot(kind="scatter", x="awards", y="norm_max_min", title="Awards vs Ratings")
    plt.savefig("ratings_vs_awards.png")
    plt.show()
    alt_plot_for_Awards_ratings(pdfg)


def alt_plot_for_Awards_ratings(pdfg):
    pdfg.plot(kind="bar", x='awards', y="ratings",title="Awards Vs Ratings")
    plt.savefig("ratings_vs_awards_alt.png")
    plt.ylabel("Mean Rating")
    plt.xlabel("Award Count")
    plt.show()
