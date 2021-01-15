![](https://tokei.rs/b1/github/jobyid/strive_build_good_reads)
[![License: WTFPL](https://img.shields.io/badge/License-WTFPL-brightgreen.svg)](http://www.wtfpl.net/about/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-no-red.svg)](https://bitbucket.org/lbesson/ansi-colors)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Website shields.io](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](http://shields.io/)


# Good Reads Best Books Analysis

An exploration of the top 1000 books on [goodreads.com](<https://goodreads.com>) to produce insights as to their qualities and produce some tools for recommending books. 

## How to use this repo:
We created 2 options, for you to explore. 
1. [Website](<https://strive-good-reads.herokuapp.com>) Here we present our findings in the clearest way and have some simple reccomendations tools for you to use. 
2. Command line tool: If you prefer to use the Terminal to access our findings you can clone the repo, and run it from your terminal. 

To get started in the terminal enter. 

`python good_reads_main.py`


You can then use the following options: 

```Options:
  -v, --visualise TEXT  Enter the name of the visulisation you would like to
                        see. Possible Options
                        ['ratings_per_year','awards_ratings', 'dis_norm_max_mi
                        n','dis_mean_norm_rating','minmax_and_mean_norm','num_
                        pages_vs_num_ratings','avg_rating_distribution',
                        'best_fit_distribution_for_avg_rating',
                        'awards_distribution','awards_boxplot']

  -s, --stats TEXT      Choose the stat representation you want to see from
                        the following options: ['bayes']

  -a, --analysis TEXT   Choose the analysis representation you want to see
                        from the following options: ['awards',
                        'original_publish_year']

  -au, --author TEXT    Enter the the name of an author in '' eg 'Jane Austen'
  -r, --recommend TEXT  Enter the last book you read and get a recommendation
                        for your next read.For book titles of more then 1 work
                        enter inside quotation marks. eg. 'Harry Potter'

  --help                Show this message and exit.
  
  ```
 
--help will display these options in your terminal. 

## Web scraping

In order to gather the data we opted for a no code tool called [Octoparse](octoparse.com). The Octoparse files we used to do the scraping are in the scrape folder above. 
What was particularly useful was the ability to clean data with Octoparse as we scraped it. Then reduced the clean and processing work later on.

## Pre processing 
Within the good_reads_preprocesing.py file is the code we used to prepare the data. That we used mfor the analysis and visualization 

## Exploring the Data     
We have prepared a website [here](<https://strive-good-reads.herokuapp.com>) which details our findings.
You can also use clone the git and use the command line to view analysis and visualisations of the data. 


