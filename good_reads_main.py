import good_reads_visulisation as grv
import sys
import click
grv.awards_ratings()
@click.command()
@click.option('--visualise','-v', help="Enter the name of the visulisation you would like to "
                                       "see. Possible Options ['ratings_per_year','awards_ratings'] ")
@click.option('--stats', '-s', help="Choose the stat reprsentation you want to see from the "
                                    "following options: ['']")

def good_reads(visualise, stats):
    if visualise == 'ratings_per_year':
        grv.ratings_per_year_joint_plot()
    if visualise == 'awards_ratings':
        grv.awards_ratings()
    print(stats)


if __name__ == '__main__':
    #print("This is main what would you like to run?")
    good_reads()
