import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import norm


# 1. Create a 2D scatterplot with `pages` on the x-axis and `num_ratings` on the y-axis.
# 2. Can you compute numerically the correlation coefficient of these two columns?

df = pd.read_csv('./data/analyse_this.csv')

def scatterplot_2d(df, x, y, xlabel, ylabel, title):
    fig, ax = plt.subplots()
    sns.set_style('darkgrid')
    sns.set(rc={'figure.figsize':(9,6)})
    ax = sns.scatterplot(x=x, y=y, data=df)
    add_label_title(xlabel=xlabel, ylabel=ylabel, title=title)
    return fig

def scatterplot_log(df, x, y, xlabel, ylabel, title):
    fig, ax = plt.subplots()
    sns.set_style('darkgrid')
    sns.set(rc={'figure.figsize':(9,6)})
    ax = sns.scatterplot(x=x, y=y, data=df)
    ax.set_yscale('log')
    add_label_title(xlabel=xlabel, ylabel=ylabel, title=title)
    return fig

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

    
def scatterplot_2d(df, x, y, xlabel, ylabel, title):
    fig, ax = plt.subplots()
    sns.set_style('darkgrid')
    sns.set(rc={'figure.figsize':(9,6)})
    ax = sns.scatterplot(x=x, y=y, data=df)
    add_label_title(xlabel=xlabel, ylabel=ylabel, title=title)
    return fig



def add_label_title(xlabel, ylabel, title):
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14);

def calc_corr_coef(df, x, y):
    return df[x].corr(df[y])



def display_distribution_hist(df, x, xlabel, ylabel, title):

    fig, ax = plt.subplots()
    ax = sns.distplot(df[x], kde=True, hist=True, label='Data Distribution',  kde_kws = {'linewidth': 2, 'legend':True})
    kde = st.gaussian_kde(df[x]) 
    idx = np.argmax(kde.pdf(df[x])) 
    plt.axvline(df[x][idx], color='red', label=f'{df[x][idx]}') 
    ax = sns.distplot(df[x], kde = False, fit=norm, norm_hist=False, hist=False, kde_kws = {'linewidth': 2, 'legend':True}, label='Normal Distribution')
    plt.legend()
    add_label_title(xlabel=xlabel, ylabel=ylabel, title=title)
    return fig 


def display_box_plot(df, x,xlabel, ylabel=None, title=None):
    fig, ax = plt.subplots()
    ax = sns.boxplot(data=df, x=x)
    add_label_title(xlabel=xlabel, ylabel=ylabel, title=title)
    return fig


def display_box_plot(df, x,xlabel, ylabel=None, title=None):
    fig, ax = plt.subplots()
    ax = sns.boxplot(data=df, x=x)
    add_label_title(xlabel=xlabel, ylabel=ylabel, title=title)
    return fig



def display_violin_plot(df, x, y=None, xlabel=None, ylabel=None, title=None):
    fig, ax = plt.subplots()
    ax = sns.violinplot(data=df, x=x, y=y, inner='quartile', scale='count')  
    add_label_title(xlabel=xlabel, ylabel=ylabel, title=title)
    plt.xticks([0,1], ['Not Series', 'Series'])
    return fig




plot_ratings_year()
awards_ratings()


scatterplot_2d =  scatterplot_2d(df, 'num_pages', 'num_ratings', 'Number of Pages', 'Number of Ratings', \
                                    'Scatter Plot Comparing \n Number of Ratings to Number of Pages of Books')
plt.show()



scatterplot_log = scatterplot_log(df, 'num_pages', 'num_ratings', 'Number of Pages', 'Number of Ratings (Log Scale)', \
                                    'Scatter Plot Comparing \n Number of Ratings(Log Scale) to Number of Pages of Books')
plt.show()


hist_fig = display_distribution_hist(df, 'avg_rating', xlabel='Average Rating', ylabel='Density', 
                       title='Frequency Distribution of Average Rating')
plt.show()


pages_ratings_corr_coef = calc_corr_coef(df, 'num_pages', 'num_ratings')

print(f'The correlation coefficient between number of pages and number of ratings is {pages_ratings_corr_coef}.')
print(f'This shows a very weak correlation between number of pages and number of ratings. This is also clearly illustrated in the scatterplot') 



box_fig = display_box_plot(df, 'avg_rating', xlabel='Average Rating',  
                       title='Box Plot Showing Distribution of Average Rating')
plt.show()


violin_fig = display_violin_plot(df, y='avg_rating', x='is_series', ylabel='Average Rating', title='Violin Plot Showing Distribution of Average Rating')
plt.show()


hist_fig = display_distribution_hist(df, 'avg_rating', xlabel='Average Rating', ylabel='Frequency', 
                       title='Frequency Distribution of Average Rating')
plt.show()




