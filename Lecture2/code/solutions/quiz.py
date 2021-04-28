# %%

import pandas as pd
import matplotlib.dates as mdates
import numpy as np
import matplotlib.pyplot as plt
import math
import datetime as dt
print('---')

# %%
# Quiz
# # 读取./data/311-service-requests.csv，探索一下这个表格
# 看看那种抱怨最多


complaints = pd.read_csv('../data/311-service-requests.csv')
complaint_counts = complaints['Complaint Type'].value_counts()
complaint_counts[:10]

# %%
# Quiz
# 看看哪个区（borouhg）抱怨马路上噪声（Noise - Street/Sidewalk）的最多
complaints[complaints['Complaint Type'] ==
           "Noise - Street/Sidewalk"]["Borough"].value_counts()

# %%
# Quiz
# 选出所有Brooklyn区和噪音有关的抱怨，提示，可以使用`.str.contains("Noise")`
is_noise = complaints['Complaint Type'].str.contains("Noise")
in_brooklyn = complaints['Borough'] == "BROOKLYN"
complaints[is_noise & in_brooklyn]
