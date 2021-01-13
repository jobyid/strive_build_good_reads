import good_reads_visulisation as grv
import good_reads_stats as grs
import sys
import click

@click.command()
@click.option('--visualise','-v', help="Enter the name of the visualisation you would like to see. Possible Options ['ratings_per_year',"
                                       "'awards_ratings', 'dis_norm_max_min',"
                                       "'dis_mean_norm_rating','minmax_and_mean_norm'] ")
@click.option('--stats', '-s', help="Choose the stat representation you want to see from the "
                                    "following options: ['bayes']")
@click.option('--analysis', '-a', help="Choose the analysis representation you want to see from "
                                       "the "
                                    "following options: ['']")


def good_reads(visualise, stats, analysis):
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

if __name__ == '__main__':
    #print("This is main what would you like to run?")
    good_reads()
