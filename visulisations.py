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
    for distribution in DISTRIBUTIONS:

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


# Plot for comparison
plt.figure(figsize=(12,8))
ax = data.plot(kind='hist', bins=50, alpha=0.5, density=True)

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
    


#scatterplot_2d =  scatterplot_2d(df, 'num_pages', 'num_ratings', 'Number of Pages', 'Number of Ratings', 'Scatter Plot Comparing \n Number of Ratings to Number of Pages of Books')
plt.show()



#scatterplot_log = scatterplot_log(df, 'num_pages', 'num_ratings', 'Number of Pages', 'Number of Ratings (Log Scale)', 'Scatter Plot Comparing \n Number of Ratings(Log Scale) to Number of Pages of Books')
plt.show()


#hist_fig = display_distribution_hist(df, 'avg_rating', xlabel='Average Rating', ylabel='Density', title='Frequency Distribution of Average Rating')
plt.show()


pages_ratings_corr_coef = calc_corr_coef(df, 'num_pages', 'num_ratings')
print(f'The correlation coefficient between number of pages and number of ratings is {pages_ratings_corr_coef}.')
print(f'This shows a very weak correlation between number of pages and number of ratings. This is also clearly illustrated in the scatterplot') 



#box_fig = display_box_plot(df, 'avg_rating', xlabel='Average Rating',  title='Box Plot Showing Distribution of Average Rating')
plt.show()


#violin_fig = display_violin_plot(df, y='avg_rating', x='is_series', ylabel='Average Rating', title='Violin Plot Showing Distribution of Average Rating')
plt.show()


#hist_fig = display_distribution_hist(df, 'avg_rating', xlabel='Average Rating', ylabel='Frequency', title='Frequency Distribution of Average Rating')
plt.show()




#plot_ratings_year()
#awards_ratings()



data = df['avg_rating']

fig1, fig2 = plot_pdf_avg_rating(data)
plt.show()