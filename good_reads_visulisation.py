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
    df.plot(kind="scatter", x="awards", y="norm_max_min", title="Awards vs Ratings")
    plt.savefig("ratings_vs_awards.png")
    plt.show()


def alt_plot_for_Awards_ratings():
    pdf = df[['norm_max_min',"awards"]]
    pdf["awards"]=pdf["awards"].fillna(0)
    pdf = pdf[pdf['awards']>0]
    pdfg = pdf.groupby(["awards"]).agg(ratings =('norm_max_min','mean'))
    pdfg.reset_index(inplace=True)
    pdfg.plot(kind="bar", x='awards', y="ratings",title="Awards Vs Ratings")
    plt.savefig("ratings_vs_awards_alt.png")
    plt.ylabel("Mean Rating")
    plt.xlabel("Award Count")
    plt.show()


#Question 4

def vis_norm_max_min(da):

    da["norm_max_min"].plot.hist( rot=0)

    plt.title('Distribution of Normalized max & min Ratings', fontsize=10)

    plt.savefig("plot_simple_histogramme_matplotlib_01.png")

    plt.show()

#Question 5

def vis_mean_norm(da):
    da["norm_mean"].plot.hist(x='books', y='normalized means', rot=0)

    plt.title('Distribution of Normalized rating means', fontsize=10)

    plt.savefig("plot_simple_histogramme_matplotlib_02.png")

    plt.show()

#Question 6

def vis_all_norm(da):
    df = da[["norm_mean","norm_max_min"]]
    df.plot.hist(rot=0)
    plt.title('Comparison of distribution of the Min/Max and Mean Norms', fontsize=10)
    plt.show()

#vis_norm_max_min(da)
#vis_mean_norm(da)
#vis_all_norm(da)




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


    
def scatterplot_2d(df):
    fig, ax = plt.subplots()
    sns.set_style('darkgrid')
    sns.set(rc={'figure.figsize':(9,6)})
    ax = sns.scatterplot(x='num_pages', y='num_ratings', data=df)
    add_label_title(xlabel='Number of Pages', ylabel='Number of Ratings (Log Scale)', title='Scatter Plot Comparing \n Number of Ratings(Log Scale) to Number of Pages of Books')
    return fig


def scatterplot_log(df):
    fig, ax = plt.subplots()
    sns.set_style('darkgrid')
    sns.set(rc={'figure.figsize':(9,6)})
    ax = sns.scatterplot(x='num_pages', y='num_ratings', data=df)
    ax.set_yscale('log')
    add_label_title(xlabel='Number of Pages', ylabel='Number of Ratings', title='Scatter Plot Comparing \n Number of Ratings(Log Scale) to Number of Pages of Books')
    return fig


def add_label_title(xlabel, ylabel, title):
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14);

def calc_corr_coef(df, x, y):
    return df[x].corr(df[y])



def display_distribution_hist(df):
    df = pd.read_csv('./data/analyse_this.csv')
    x ='avg_rating'
    xlabel = 'Average Rating'
    ylabel='Density'
    title = 'Frequency Distribution of Average Rating'
    fig, ax = plt.subplots()
    ax = sns.distplot(df[x], kde=True, hist=True, label='Data Distribution',  kde_kws = {'linewidth': 2, 'legend':True})
    kde = st.gaussian_kde(df[x]) 
    idx = np.argmax(kde.pdf(df[x])) 
    plt.axvline(df[x][idx], color='red', label=f'Maximum: {df[x][idx]}') 
    ax = sns.distplot(df[x], kde = False, fit=st.norm, norm_hist=False, hist=False, kde_kws = {'linewidth': 2, 'legend':True}, label='Normal Distribution')
    plt.legend()
    add_label_title(xlabel=xlabel, ylabel=ylabel, title=title)
    return fig 


def display_box_plot(df):
    fig, ax = plt.subplots()
    ax = sns.boxplot(data=df, x='avg_rating')
    add_label_title(xlabel='Average Rating', ylabel='Frequency', title='Distribution of Average Rating')
    return fig




def display_violin_plot(df):
    fig, ax = plt.subplots()
    ax = sns.violinplot(data=df, x='is_series', y='avg_rating', inner='quartile', scale='count')  
    add_label_title(xlabel='Frequency', ylabel='Average Rating', title='Distribution of Average Rating By Series')
    plt.xticks([0,1], ['Not Series', 'Series'])
    return fig





plt.rcParams['figure.figsize'] = (16.0, 12.0)
plt.style.use('ggplot')

# Create models from data
def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check
    DISTRIBUTIONS = [        
        st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.cauchy,st.chi,st.chi2,st.cosine,
        st.dgamma,st.dweibull,st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk,
        st.foldcauchy,st.foldnorm,st.frechet_r,st.frechet_l,st.genlogistic,st.genpareto,st.gennorm,st.genexpon,
        st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,
        st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,
        st.invweibull,st.johnsonsb,st.johnsonsu,st.ksone,st.kstwobign,st.laplace,st.levy,st.levy_l,st.levy_stable,
        st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,st.ncx2,st.ncf,
        st.nct,st.norm,st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.reciprocal,
        st.rayleigh,st.rice,st.recipinvgauss,st.semicircular,st.t,st.triang,st.truncexpon,st.truncnorm,st.tukeylambda,
        st.uniform,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy
    ]

    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    # Estimate distribution parameters from data
    for distribution in DISTRIBUTIONS[:3]:

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)
                    end
                except Exception:
                    pass

                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse

        except Exception:
            pass

    return (best_distribution.name, best_params)



def make_pdf(dist, params, size=10000):
    """Generate distributions's Probability Distribution Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf

df = pd.read_csv('./data/analyse_this.csv')

data = df['avg_rating']

def plot_avg_best_distribution(data):
    # Plot for comparison
    plt.figure(figsize=(12,8))
    ax = data.plot(kind='hist', bins=50, alpha=0.5, density=True)
    #sns.distplot(pdf, fit=norm, hist=True)

    # Save plot limits
    dataYLim = ax.get_ylim()

    # Find best fit distribution
    best_fit_name, best_fit_params = best_fit_distribution(data, 200, ax)
    best_dist = getattr(st, best_fit_name)

    # Update plots
    ax.set_ylim(dataYLim)
    ax.set_title(u'Average Rating of The Best Books Ever\n All Fitted Distributions')
    ax.set_xlabel(u'Average Rating')
    ax.set_ylabel('Frequency')

    # Make PDF with best params 
    pdf = make_pdf(best_dist, best_fit_params)

    # Display
    plt.figure(figsize=(12,8))
    ax = pdf.plot(lw=2, label='PDF', legend=True)

    data.plot(kind='hist', bins=50, alpha=0.5, label='Data', legend=True,density=True, ax=ax)

    param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
    param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_fit_params)])
    dist_str = '{}({})'.format(best_fit_name, param_str)

    ax.set_title(u'Average Rating of The Best Books Ever with best fit distribution \n' + dist_str)
    ax.set_xlabel(u'Average Rating')
    ax.set_ylabel('Frequency')
    plt.legend()
    plt.show()

