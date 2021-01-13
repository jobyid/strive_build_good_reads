import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# https://stackoverflow.com/questions/33468976/pandas-conditional-probability-of-a-given-specific-b
def bayes_prop():
    dfs = pd.read_csv('data/review_ratings_series.csv')
    df = pd.read_csv('data/analyse_this.csv')
    s = dfs[['series','title']]
    s.replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"], value=["",""], regex=True, inplace=True)
    s.title = s['title'].str.strip()
    a = df[['awards','title']].fillna(0)
    f = pd.merge(a,s)
    # find P(A|B)
    pab = pd.crosstab(f.awards, f.series, normalize='columns')
    print(pab)
    pab.plot(kind='bar', title="Probaility of Award count given Series ' \
                                               'or not")
    plt.show()
    return  pab

