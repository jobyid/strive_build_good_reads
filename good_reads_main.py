import good_reads_visulisation as grv
import good_reads_stats as grs
import sys
import click
import pandas as pd
import matplotlib.pyplot as plt

@click.command()
@click.option('--visualise','-v', help="Enter the name of the visulisation you would like to "
                                       "see. Possible Options ['ratings_per_year',"
                                       "'awards_ratings', 'dis_norm_max_min',"
                                       "'dis_mean_norm_rating','minmax_and_mean_norm','num_pages_vs_num_ratings',"
                                       "'avg_rating_distribution', 'best_fit_distribution_for_avg_rating'] ")
@click.option('--stats', '-s', help="Choose the stat representation you want to see from the "
                                    "following options: ['bayes']")
@click.option('--analysis', '-a', help="Choose the analysis representation you want to see from "
                                       "the "
                                    "following options: ['']")




def good_reads(visualise, stats, analysis):
    df = pd.read_csv('./data/analyse_this.csv')
    if visualise == 'ratings_per_year':
        grv.ratings_per_year_joint_plot()
    if visualise == 'awards_ratings':
        grv.awards_ratings()
        grv.alt_plot_for_Awards_ratings()
    if visualise == 'dis_norm_max_min':
        grv.vis_norm_max_min(grv.df)
    if visualise == 'dis_mean_norm_rating':
        grv.vis_mean_norm(grv.df)
    if visualise == 'minmax_and_mean_norm':
        grv.vis_all_norm(grv.df)
    if stats == 'bayes':
        grs.bayes_prop()
    if visualise == 'num_pages_vs_num_ratings':
        grv.scatterplot_2d(df)
        grv.scatterplot_log(df)
        plt.show()
    if visualise == 'avg_rating_distribution':
        grv.display_distribution_hist(df)
        grv.display_box_plot(df)
        grv.display_violin_plot(df)
        plt.show()
    if visualise == 'best_fit_distribution_for_avg_rating':
        data = df['avg_rating']
        grv.plot_avg_best_distribution(data)
    if len(sys.argv[1:]) < 1:
        click.echo('Please enter an option after the file name to find the available options use --help')


if __name__ == '__main__':
    #print("This is main what would you like to run?")
    good_reads()
    #Question 1



#Question 2
pages_ratings_corr_coef = grv.calc_corr_coef(df, 'num_pages', 'num_ratings')
print(f'The correlation coefficient between number of pages and number of ratings is {pages_ratings_corr_coef}.')
print(f'This shows a very weak correlation between number of pages and number of ratings. This is also clearly illustrated in the scatterplot') 
