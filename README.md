![](https://tokei.rs/b1/github/jobyid/strive_build_good_reads)
[![License: WTFPL](https://img.shields.io/badge/License-WTFPL-brightgreen.svg)](http://www.wtfpl.net/about/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-no-red.svg)](https://bitbucket.org/lbesson/ansi-colors)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Website shields.io](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](http://shields.io/)

[See the work here](<>)
#Good Reads Best Books Analysis

In this project we were set with the task of scraping data from the goodreads.com and then performing some analysis of the data we gathered. 

##Web scraping 
In order to gather the data we opted for a no code tool called [Octoparse](octoparse.com). The Octoparse files we used to do the scraping are in the scrape folder above. 
What was particularly useful was the ability to clean data with Octoparse as we scraped it. Then reduced the clean and processing work later on.

##Pre processing 
Within the good_reads_preprocesing.py file is the code we used to prepare the data. That we used mfor the analysis and visualization 

##Exploring the Data     
We have prepared a website here which details our findings.
You can also use clone the git and use the command line to view analysis and visualisations of the data. 

To use the the command line: 
1. change directory to the folder of the cloned git. 
2. enter `python good_reads_main.py --help` to get the list of options. 
3. enter 'good_reads_analyse.py' to understand more about the data
