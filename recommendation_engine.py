

import pandas as pd #line:1
import random #line:2
def recoomend_a_book ():#line:4
    OO00O00OOO0OO00OO =pd .read_csv ('data/good_reads_df_web.csv')#line:5
    OOO0OO0O00000OO0O =OO00O00OOO0OO00OO ['Title']#line:6
    OO0OO00O0OOOOOO0O =random .choice (OOO0OO0O00000OO0O )#line:7
    OO00O0000OOOO0O00 =OO00O00OOO0OO00OO [OO00O00OOO0OO00OO ['Title']==OO0OO00O0OOOOOO0O ]#line:8
    return OO00O0000OOOO0O00 ['Title'].values [0 ]+" by "+OO00O0000OOOO0O00 ['Author'].values [0 ]
