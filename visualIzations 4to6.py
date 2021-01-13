import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

da = pd.read_csv('data/analyse_this.csv')

def vis_norm_max_min(da):

    da["norm_max_min"].plot.hist( rot=0)

    plt.title('Normalized max & mins', fontsize=10)

    plt.savefig("plot_simple_histogramme_matplotlib_01.png")

    plt.show()

def vis_mean_norm(da):
    da["norm_mean"].plot.hist(x='books', y='normalized means', rot=0)

    plt.title('Normalized means', fontsize=10)

    plt.savefig("plot_simple_histogramme_matplotlib_02.png")

    plt.show()

def vis_all_norm(da):

    df = da[["norm_mean","norm_max_min"]]
    df.plot.hist(rot=0)
    plt.show()

vis_norm_max_min(da)
vis_mean_norm(da)
vis_all_norm(da)
