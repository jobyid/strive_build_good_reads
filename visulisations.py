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

def scatterplot_2d(df, x, y, xlabel, ylabel, title):
    scatter_fig = plt.figure()
    sns.set_style('darkgrid')
    sns.set(rc={'figure.figsize':(9,6)})
    sns.scatterplot(x=x, y=y, data=df)
    add_label_title(xlabel=xlabel, ylabel=ylabel, title=title)
    return scatter_fig


def add_label_title(xlabel, ylabel, title):
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=16);


def calc_corr_coef(df, x, y):
    return df[x].corr(df[y])



def display_distribution_hist(df, x, xlabel, ylabel, title):
    hist_fig = plt.figure()
    df[x].plot(kind='hist')
    add_label_title(xlabel=xlabel, ylabel=ylabel, title=title)
    return hist_fig 


def display_box_plot(df, x,xlabel, ylabel=None, title=None):
    sns.boxplot(data=df, x=x)
    add_label_title(xlabel=xlabel, ylabel=ylabel, title=title)
    plt.show()



def display_violin_plot(df, x, y=None, xlabel=None, ylabel=None, title=None):
    sns.violinplot(data=df, x=x, y=y, inner='quartile', scale='count')  
    add_label_title(xlabel=xlabel, ylabel=ylabel, title=title)
    plt.xticks([0,1], ['Not Series', 'Series'])
    plt.show()






df = pd.read_csv('./data/analyse_this.csv')

scatterplot_2d =  scatterplot_2d(df, 'num_pages', 'num_ratings', 'Number of pages', 'Number of ratings', \
                                    'Scatter Plot Comparing \n Number of Ratings to Number of Pages of Books')
plt.show()


pages_ratings_corr_coef = calc_corr_coef(df, 'num_pages', 'num_ratings')

print(f'The correlation coefficient between number of pages and number of ratings is {pages_ratings_corr_coef}.')
print(f'This shows a very weak correlation between number of pages and number of ratings. This is also clearly illustrated in the scatterplot') 



box_fig = display_box_plot(df, 'avg_rating', xlabel='Average Rating',  
                       title='Box Plot Showing Distribution of Average Rating')
plt.show()



hist_fig = display_distribution_hist(df, 'avg_rating', xlabel='Average Rating', ylabel='Frequency', 
                       title='Frequency Distribution of Average Rating')
plt.show()



violin_fig = display_violin_plot(df, y='avg_rating', x='is_series', ylabel='Average Rating', title='Violin Plot Showing Distribution of Average Rating')
plt.show()