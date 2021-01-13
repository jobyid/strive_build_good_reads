import good_reads_visulisation as grv




#Question 2
pages_ratings_corr_coef = grv.calc_corr_coef(df, 'num_pages', 'num_ratings')
print(f'The correlation coefficient between number of pages and number of ratings is {pages_ratings_corr_coef}.')
print(f'This shows a very weak correlation between number of pages and number of ratings. This is also clearly illustrated in the scatterplot') 




