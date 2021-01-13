import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def vis_norm_max_min(da):

    ax = da["norm_max_min"].plot.bar(x='books', y='normalized values', rot=0)

    plt.hist(ax)

    plt.title('Normalized max & mins', fontsize=10)

    plt.savefig("plot_simple_histogramme_matplotlib_01.png")

    plt.show()

def vis_mean_norm(da):
    ax = da["norm_mean"].plot.bar(x='books', y='normalized means', rot=0)
    plt.hist(ax)

    plt.title('Normalized means', fontsize=10)

    plt.savefig("plot_simple_histogramme_matplotlib_02.png")

    plt.show()

def vis_all_norm(da):
    mean = da["norm_mean"]
    maxmin = da["norm_max_min"]

<<<<<<< Updated upstream
    df = pd.DataFrame({'mean': mean,
                       'maxmin': maxmin}, index=index)
    ax = df.plot.bar(rot=0)
=======
    df = da[["norm_mean","norm_max_min"]]
    df.plot.bar(rot=0)
    plt.show()

vis_norm_max_min(da)
vis_mean_norm(da)
vis_all_norm(da)
>>>>>>> Stashed changes
