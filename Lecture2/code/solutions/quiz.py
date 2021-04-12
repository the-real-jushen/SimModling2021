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

# %%

# 练习，画出covid-19 confiremed case变化趋势和增长率曲线，只画出确诊累计人数最多的5个国家

cases = pd.read_csv("../covid-19-cases.csv")
# %%
cases["sum"] = np.sum(cases.iloc[:, 5:-1], 1)
casesByCounty = cases.drop(columns=cases.columns[[0, 1, 3, 4]])
countryRegions = casesByCounty["Country/Region"].unique()

casesByCounty = casesByCounty.groupby('Country/Region').sum()

topCases = casesByCounty.sort_values("sum")[-5:]

dates = [dt.datetime.strptime(date, u'%m/%d/%y').date()
         for date in topCases.columns[:-1].to_numpy()]

ax = plt.subplot(211)
for i in range(topCases.index.shape[0]):
    plt.plot(dates, topCases[i:i +
             1].to_numpy().reshape(-1)[:-1], label=topCases.index[i])


years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
years_fmt = mdates.DateFormatter('%Y')
# Text in the x axis will be displayed in 'YYYY-mm' format.
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
# format the ticks
# Major ticks every 6 months.
fmt_half_year = mdates.MonthLocator(interval=6)
ax.xaxis.set_major_locator(fmt_half_year)

# Minor ticks every month.
fmt_month = mdates.MonthLocator()
ax.xaxis.set_minor_locator(fmt_month)


plt.legend()


ax = plt.subplot(212)
for i in range(topCases.index.shape[0]):
    plt.plot(dates[1:], np.diff(topCases[i:i+1].to_numpy()
                                ).reshape(-1)[:-1], label=topCases.index[i])
ax.xaxis.set_major_locator(fmt_half_year)
ax.xaxis.set_minor_locator(fmt_month)

plt.legend()


# %%
