# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 13:16:41 2020

@author: MAHNAPA
"""
#%%

"""
 +++++++++++ One hot Encoding ++++++++++++++

It is a process of converting categorical variables
into features or columns and coding one or zero for the presence of that
particular category
+++++++++++ One hot Encoding ++++++++++++++

"""

import pandas as pd

text = " i love leraning NLP"

pd.get_dummies(text.split())

#%%


