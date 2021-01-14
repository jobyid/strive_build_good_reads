import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import good_reads_analyse as gra
import time

st.subheader("Scraping the Data")
st.markdown("The data for our exploration of the best reads came from the website [Good Reads]("
            "<https://www.goodreads.com>), unfortunately the API for the website has been "
            "discontinued, so we scrapped the data. To do this we used [Octoparse]("
            "<https://octoparse.com>). Each of the Octoparse scripts we used are [here]("
            "<https://github.com/jobyid/strive_build_good_reads/tree/main/Scrape>) one of the "
            "exciting things about using Octoparse was the bility to clean the data as the "
            "scraping process happened. In the end we were left with a string of csv files")


