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

#%% [markdown]
""" 
一个小球，在地球上水平抛出，初始高度为h，速度为v，落地后弹起时，
垂直速度为原来的0.7水平速度不变，请画出他在第二次落地前的轨迹

增加一点难度，如果起的最高点小于m，这次落地以后就结束绘图
 """

#%%
def ball(h, v,m):
    g = 9.8
    d=0
    plt.axhline(y=m, color='r', linestyle='--') 
    while True:
        if d==0:
            # the initial drop
            fallTime = (2*h/g)**0.5
            v2 = fallTime*0.7*g
        else:
            # the bounce
            fallTime=v2/g*2
            
        # the horizontal move
        h_dist=v*fallTime
        x1 = np.linspace(0, v*fallTime, 20)
        yTime = x1/v
        
        if d==0:
            # the initial drop, y coordinates at x time
            y1 = h-0.5*g*yTime**2
        else:
            # bounce, free fall with positive initial speed at ground level,
            # y coordinates at x time
            y1 = yTime*v2-0.5*g*yTime**2
            h=np.max(y1)
            v2 = v2*0.7
        plt.plot(x1+d, y1)
        d=d+h_dist
        if h<m:
            break


ball(10, 10,2)

# %%
