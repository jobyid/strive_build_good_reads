import good_reads_analyse as gra
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import recommendation_engine as re
import time
df = pd.read_csv('data/good_reads_df_web.csv',index_col=0)
da = pd.read_csv('data/awards_by_year.csv')

b1, b2 = st.beta_columns([1,8])
with b1:
    st.markdown('[![](<https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github'
            '&logoColor=white>)](<https://github.com/jobyid/strive_build_good_reads>)')
#python bagde
with b2:
    st.markdown('![](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo'
                     '=python'
            '&logoColor=white)')


st.title("Explore our collection!")

st.header("Reading. Data Science supported.")
st.write(" ")
st.subheader("Are you looking for a really good read that is worth your while? Think no more, choose one of the following from our cross checked database.")
st.write(" ")
st.image("fig/12.jpg", use_column_width = True)
st.write(" ")

st.subheader("Top 1000 books recommended by top readers")
st.dataframe(df)
st.write(" ")

# table
st.sidebar.title("1000 Good reads")
st.sidebar.write("More time reading. Less time searching")

st.sidebar.image("fig/5231.jpg", use_column_width = True)

st.write(" ")
st.write("In 1000 Good Reads we believe that reading should not be a complicated, stressful or even an unsatisfying hobbie."
         " If you are not certain on where to put your money, you can **relax** and trust on our Data Science team."
         " We are there to pre-select the **best reads**")
st.write(" ")
ballons = False
with st.sidebar.beta_container():
    st.markdown("**Do you need a book recommendation?**")
    st.markdown("""Well why not let some science help you, use our book recommendation engine 
    below, and get the perfect next read. """)
    st.text_input("Enter the last book you read:")
    if st.button("Recommend"):
        st.write("**The science says you should read:**")
        with st.spinner("Wait For it"):
            time.sleep(2)
            ballons = True
        book = re.recoomend_a_book()
        st.markdown(book)
if ballons:
    st.balloons()
with st.beta_container():
    st.subheader("Representing the Data")
    col1, col2, col3 = st.beta_columns(3)
    with col1:
        st.markdown("**Fit a Distribution**")
        with st.beta_expander("See explanation"):
            st.write("""
                This chart shows how our the average rating of the best books ever
                 fits to the full list of the current scipy.stats distributions  
                 and also determine the distribution with the least error. """)
        st.image('fig/all_distributions.png',use_column_width=True)

        st.write("")
        st.write("")
        st.markdown("**Highest Rated Books**")
        with st.beta_expander("See explanation"):
            st.write("A representation of the top 10 highest rated books")
        st.image('fig/Best_Books_With_Highest_Rating.png',use_column_width=True)

        st.write("")
        st.write("")
        st.markdown("**Ratings Vs Series**")
        with st.beta_expander("See explanation"):
            st.write("""
                The chart shows the relationship between average rating and if a book is 
                part of a series or not. As you can see from the figure there is not a 
                significant differance. In fact from this data we concluded that whether or not a book is in a series of books did not affect its rating.   
                """)
        st.image('fig/Avg_Rating_Violin_plot.png',use_column_width=True)
    with col2:
        st.markdown("**Distribution**")
        with st.beta_expander("See explanation"):
            st.write("""
                The chart depicts how closely distribution of average rating of books rated as best ever
                follows that of a normal distribution and that the highest point on the data distribution curve is 4.03.
                 This implies that most books had a rating around 4.03.
                """)
        st.image('fig/avg_distribution.png',use_column_width=True)

        st.write("")
        st.write("")
        st.markdown("**Ratings vs Year**")
        with st.beta_expander("See explanation"):
            st.write("Showing the ratings vs the year ")
        st.image('fig/ratings_year_joint.png',use_column_width=True)

        st.write("")
        st.write("")
        st.markdown("**Highest Reviewed Books **")
        with st.beta_expander("See explanation"):
            st.write("The top 10 highest reviewed books")
        st.image("fig/Best_Books_With_Highest_Reviews.png",use_column_width=True)
    with col3:
        st.markdown("**Average Rating**")
        with st.beta_expander("See explanation"):
            st.write("""
                The chart illustrates how 50% of average ratings of books lie between 3.95 to 4.21.
                We can also see that there are a few outliers.
                """)
        st.image('fig/Avg_Rating_boxplot.png',use_column_width=True)

        st.write("")
        st.write("")
        st.markdown("**Books with most Awards**")
        with st.beta_expander("See explanation"):
            st.write("top 10 ten most awarded books")
        st.image("fig/Best_Books_With_Most_Awards.png",use_column_width=True)

        st.write("")
        st.write("")
        st.markdown("**Pages vs Ratings**")
        with st.beta_expander("See explanation"):
            st.write("Number of pages vs rating")
        st.image("fig/num_pages_rating_scatter.png",use_column_width=True)



with st.sidebar.beta_container():
    st.subheader("What's an Authors Best Rated Book? Find Out Below")
    auth1, auth2 = st.beta_columns([3, 1])
    with auth1:
        title = st.text_input('Enter an Author Name to find their best rated book')
    with auth2:
        st.write("")
        st.write("")
        search = st.button("Search")
    if search:
        st.write(title, "'s best rated book is: ",gra.my_best_book(title))
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")
st.write(" ")

#st.write("So,what are you waiting for?   **Subscribe Now!**   and get a **30% discount** on your first month. ")

#Our collection contains an average rating of 4 to 4.2 stars with a certainty of 70% . So rest assured you'll be getting a top rated book-feed without taking the effort of researching.

#Graphs awards-ratings

#Despite the awards count, you can rely on us to select good quality books for your mind.
st.title("Observations/Conclusions")
st.write("So here is what we can conclude from all our work.")
st.markdown('''1. New awards are invented each each which is why newer books have mre awards.
2. The best measure of a book is the number of ratings, this offer the most significant 
differance between best and worse. 
3. Average rating is not a good measure as it seems there is a confirmation biases where most 
people rate books within a smaller range to the current average. with very few outliers. We 
Suggests this is because readers feel unable to leave a rating widely outside the current range.
4. All the top 1000 books in the good read best book ever list are good books, you are unlikely 
to go wrong with any of them. 
5. Given that all the books on the list are good this may suggest that exploring books outside 
the genre you typically read would be fruitful, and enhance your reading experience. ''')

st.title("The Science behind it all")
st.subheader("Scraping the Data")
st.markdown("The data for our exploration of the best reads came from the website [Good Reads]("
            "<https://www.goodreads.com>), unfortunately the API for the website has been "
            "discontinued, so we scrapped the data. To do this we used [Octoparse]("
            "<https://octoparse.com>). Each of the Octoparse scripts we used are [here]("
            "<https://github.com/jobyid/strive_build_good_reads/tree/main/Scrape>) one of the "
            "exciting things about using Octoparse was the bility to clean the data as the "
            "scraping process happened. In the end we were left with a string of csv files")
st.subheader("Processing the Data")
st.markdown("With our freshly created set of CSV files we need to work some pandas magic and "
            "wrangle this into something useful. Cleaning data was key, although much of this had been done in scraping, then we needed to make some dataframes for analysis. We wanted to explore how the ratings played a role im the rankings, so we performed some normalisations on the ratings data.")
st.markdown("The code we used for this process can be found [here]("
            "<https://github.com/jobyid/strive_build_good_reads/blob/main/good_reads_preprocessing.py>) ")

with st.beta_expander("See the Code"):
    st.code('''import pandas as pd
import numpy as np




def pre_process(csv_filepath1, csv_filepath2, csv_filepath3, csv_filepath4):

    df = pd.read_csv(csv_filepath1, index_col=0)
    df1 = pd.read_csv(csv_filepath2)
    df2 = pd.read_csv(csv_filepath3)
    df3 = pd.read_csv(csv_filepath4)

    df1.dropna(subset=['original_publish_year'], inplace=True)

    publish_years = df1['original_publish_year'].astype('int').values[:900]

    title = df1['title'].apply(lambda x:x.strip())[:900].values

    awards = df1['awards'].str.split(',').agg(np.size).astype('int').replace(1, np.nan).values[:900]

    avg_ratings = df['minirating'].str.split('avg').apply(lambda x: x[0])
    avg_ratings =  avg_ratings.str.extract('(\d.+)')[0].apply(pd.to_numeric)[:900].values

    num_rating = df['minirating'].str.split('avg').apply(lambda x: x[1].split(' ')[3])
    num_rating = num_rating.str.split(',').apply(lambda x: ''.join(x)).apply(pd.to_numeric)[:900].values

    author = df['Author'][:900].values

    genres = df3['genre']

    locations = df3['locations']

    num_pages = df3['num_pages']

    num_reviews = df2['num_reviews'][:900]
    #num_rating_f = df['num_rating'][:900]
    is_series = df2['series'][:900]

    #title = df['Title']

    url = df['Title_URL'][:900].values

    data = {'url': url, 'title': title, 'author': author, 'num_ratings': num_rating,
            'avg_rating': avg_ratings, 'awards': awards, 'original_publish_year': publish_years,
            'num_reviews': num_reviews, 'is_series': is_series, 'genre': genres, 'location': locations, 'num_pages': num_pages}

    good_read = pd.DataFrame(data)


    return good_read


def mean_minmax_normalisation(df):
    da = df["avg_rating"]
    normalized_da =  1 + ((da - da.mean())/(da.max() - da.min())) * 9
    normalized_df_max_min = 1 + ((da - da.min())/(da.max() - da.min())) * 9
    df["norm_mean"] = normalized_da
    df["norm_max_min"] = normalized_df_max_min
    return df

def clean_the_places():
    df = pd.read_csv('data/places.csv')
    df.replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"], value=["",""], regex=True, inplace=True)
    places = df['places'].dropna()

    print(places)


''')
st.subheader("Analyse the Data")
st.markdown("The next step was to do some anaylsis of the data and create some useful tools. The "
            "best book by an author came from this step, along with producing some insights in "
            "what to visulise.")

with st.beta_expander("See the code"):
    st.code('''import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    df = pd.read_csv('data/analyse_this.csv')

    def my_best_book(author):
        if df['author'].str.contains(author).sum() > 0:
            autors_books = df[df['author']== author]
            rating = autors_books.loc[autors_books['norm_max_min'].idxmax()]
            return rating.title
        return "Author not found"

    def awards(df):
        data=df
        df = data.groupby("awards")["awards"].count()
        print(df)


    def original_publish_year(df):
        data=df
        dfs = data.groupby("original_publish_year")["num_ratings"].mean()
        print(dfs)''')

st.write("")
st.write("")
st.subheader("Visualise the Data")
st.markdown("Then came the fun part, visualising the data. You have seen a selection of the "
            "graphs we produced above. Below is the code we used to get them.")

with st.beta_expander("See the Code"):
    st.code('''import numpy as np
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
    plt.savefig("fig/ratings_year_joint.png")
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

#df = pd.read_csv('./data/analyse_this.csv')

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


    #Question number 8

def awards_distribution(df):
    data = df
    count = data["awards"].value_counts().values

    index = data["awards"].value_counts().index

    #data.plot.bar(x = "awards", y = count)
    plt.bar(index,count)
    plt.title("award distribution in bars")
    plt.xlabel("awards")
    plt.ylabel("Number of books")
    plt.show()

def awards_boxplot(df):

    da = df.boxplot(column = ["awards"], grid = False, color = "#0000ff");

    plt.title("boxplot for awards");
    plt.suptitle("");
    plt.show()

''')

st.write("")
st.write("")
st.write("")
st.write("")
st.write("")

st.subheader("Interesting Data")
st.dataframe(da)


with st.sidebar.beta_container():
    st.subheader("The Team of Book Readers and Data Lovers")
    team1, team2 = st.beta_columns(2)
    with team1:
        st.markdown("[**Joby ingram-Dodd**](<https://github.com/jobyid>)")
        st.image("fig/joby.png",use_column_width=True)
        st.markdown("[**Ibrahim Animashaun**](<https://github.com/iaanimashaun>)")
        st.image("fig/ibrahim.png",use_column_width=True)
    with team2:
        st.markdown("[**Martin Vilar Karlen**](<https://github.com/mvilar2018>)")
        st.image("fig/martin.png",use_column_width=True)
        st.markdown("[**Oluseyi Oyedemi**](<https://github.com/Seyi85>)")
        st.image("fig/oluseyi.png",use_column_width=True)

with st.sidebar.beta_container():
    st.subheader("Like what we are doing? Why not buy us a coffee?")
    st.markdown('[![](<https://www.buymeacoffee.com/library/content/images/2020/09/image--67--1.png>)]('
            '<https://www.buymeacoffee.com/joby>)')
