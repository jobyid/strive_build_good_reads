import pandas as pd

from pre_processing1.py import pre_process1

def pre_pro2():
    da = df["avg_rating"]
    normalized_da =  (1 + (da - da.mean())/ (da.max() - da.min()) * 9
    normalized_df_max_min = 1 + (da - da.min())/ (da.max() - da.min()) * 9
    da["norm_mean"] = normalized_da
    da["norm_max_min"] = normalized_df_max_min
    return normalized_df
    return normalized_df_max_min
