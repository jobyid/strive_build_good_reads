import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns

# 1. Create a 2D scatterplot with `pages` on the x-axis and `num_ratings` on the y-axis.
# 2. Can you compute numerically the correlation coefficient of these two columns?



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

print(f'The correlation coeficient between number of pages and number of ratings is {pages_ratings_corr_coef}. This shows a very weak correlation between number of pages and number of ratings')



box_fig = display_box_plot(df, 'avg_rating', xlabel='Average Rating',  
                       title='Box Plot Showing Distribution of Average Rating')
plt.show()



hist_fig = display_distribution_hist(df, 'avg_rating', xlabel='Average Rating', ylabel='Frequency', 
                       title='Frequency Distribution of Average Rating')
plt.show()



violin_fig = display_violin_plot(df, y='avg_rating', x='is_series', ylabel='Average Rating', title='Violin Plot Showing Distribution of Average Rating')
plt.show()
