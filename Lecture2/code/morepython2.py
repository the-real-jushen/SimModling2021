# %% [markdown]
"""
## 这里我们学习如何读取数据文件

我们这里讲一下简单的数据文件的读取，我们这里仅仅只讲csv文件，其实很多别的文件都一样，都是用pandas读取进来变成一个data frame。
虽然data frame这个概念很重要，但是我们就不展开讲了，大家理解就是data frame就是一个类似于excel表格一样的东西。
通过pandas可以把csv，excel，mat等等都读取成为一个data frame

这次课中我们只需要使用pandas，读取数据，还有numpy简单操作一下数据，matplotlib画个图，所以不需要import太多东西。
"""

# %%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
print("------")

# %% [markdown]
"""
## 基本的读取

这里有个假的数据`data.csv`。你可以用excel打开它，如下图。csv就是逗号分隔符文件，就是一个文本的表格。实际上比一定是逗号分割，
你可以在padans读取的时候，指定分隔符。

![](2021-03-28-00-37-02.png)

大家亲参考下面的注释看看基本的操作。

这里要说以西表格的index，数组值能够用数字来index，而df可以用各种变量来访index，翻遍你来选择哪一行。

大家可以用variable explorer看看都进来的df长什么样子。
"""

# %%
# 读取文件，默认都进来时一个data frame
df = pd.read_csv('data.csv')
# 这可以打印前5行出来，你可以看到这个表格的表头还有index
print(df.head())
# 这里显示基本的信息，比如有多少行多少列
print(df.info())
# %%
# 显示这个表格的每一列的基本统计信息，最大值最小值方差什么的，
df.describe()
# %%
# 可以把一个表格转成一个numpy array
dfarray = df.to_numpy()
# %%
# 我可以把表格转置，这样表头就变成index了
df2 = df.T
df2.index

# %%[markdown]
'''
## 选择数据
我们现在来选择表格里面的数据。亲根据下面的注释学习。

总的来说你和一个2位的np array很像，

DF最重要的两个属性就是`loc`和`iloc`,`loc`是用来用index还有column的名字来选择的，
iloc则是和数组一样通过坐标连选择的。要注意，loc这样读出来的数据不再是dataframe，
而实series，和df很像，也是pandas的一种数据结构
'''
# %%
# selecting data，使用loc可以读取指定行
df.loc[1]

# %%
# 也可读取一个范围，注意这时index的范围，如果你的index不是数字时字符串或者别的也可以的，就是两个index之间的所有行都被读取了
# 要注意的是，这个范围时前后都包含了的，下面就会读出第0行到第5行，一共6行
df.loc[:5]

# %%
# 也可以读取一个或多个表格单元
df.loc[5, "Pulse"]
df.loc[5, ["Pulse", "Duration"]]

# %%
# 你可以完全像数组一样,用坐标操作df，找你要使用iloc就行，要注意一点塔的输入不支持tuple，支持list
df.iloc[2:5, [0, 1, -1]]

# %%
# 原来的表里面没有index，所以都进来的时候自动用数字当作index了。
# index是可以重复的，我这里把make这一列设置为index
df2 = df.set_index("make")
# 然后我选择一个index
df2.loc["honda"]

# %%
# 偷懒的画下面这种方式也可以额
# 选择第一行，你可以改改看，选择别的行，注意注意，不能直接像list那样指定某一行，只能指定一个范围
df[1:2]


# %%
# 选择某一列
df['Pulse']
# %%
# 选择这里列满足某个条件的所有行
df[df['Pulse'] == 110]


# %%
# 选择前4行，指定的两列
df[:4][['Pulse', 'Duration']]

# %%
# 这个是一个基本的统计操作，她把某一列里面，同样的值的个数计算出来。就比方说，下面Pulse=100的行有19个。
df.value_counts("Pulse")

# %%
# unique
# 找到某一列所有的独立的值
df["Pulse"].unique()

# %%
# append row
# 添加一行也很简单，只要先用index选择一行，然后给他一个list就可以
df.loc[180] = [180, 180, 180, 180, "mazda"]
# 也可以给一个字典，这样就不用给出每一列了
df.loc[182] = {"Pulse": 181}

# %%
# 你也可以做一个df然后插入到另一个df种，注意默认会保持新的df的index，如果喝酒的冲突了就会很麻烦，
# 有一个简单的方法，就是设置`ignore_index`，这样他就会生成新的index不和以前的重复
dfToAppend = pd.DataFrame([[111, 222]], columns=('Pulse', 'Duration'))
# 你可以把ignore_index改成false试试
df.append(dfToAppend, ignore_index=True)

# %%
# 你也可以增加一列，分别指定在什么位置插入、列名，还有新的值，这个如果是list，长度与要等与行数
# 后面是是否允许重复列
df.insert(4, "sum", np.zeros((df.shape[0])), True)
df["sum"] = np.sum(df[['Pulse', 'Duration']], 1)
# 下面这一行和上面是等价的
df.insert(4, "sum", np.sum(df[['Pulse', 'Duration']], 1), True)

# %%
# drop column
# 当然你也可以删掉一列
df.drop(columns=["sum"])

# %%
# sort
# 你也可以根据某一列来排序，inplace就是说就在原来的数据上做修改，如果inplace=False，
# 原数据不会变，而实返回一个新的DF
df.sort_values('Pulse', inplace=True)
df[:3]
# %%[markdown]
'''
## 简单的数据清理 
有时候数据都进来不太干净，需要清理一下，去掉一些重复的或着不是数据的行或者列。

'''
# %%
# clean data
# Empty cells
# Duplicates

# 删掉含有非数字的行
df.dropna(inplace=True)
df.info()

# %%
# 或者是对非数字的单元格填充上一个固定的数字，
df = pd.read_csv('data.csv')
df.fillna(130, inplace=True)
df.info()
# %%
# 有时候会用中位值或者平均值代替缺失的数
df = pd.read_csv('data.csv')
df['Calories'].fillna(df['Calories'].mean(), inplace=True)
df.info()

# %%
dup = df.duplicated().to_numpy()
dup.sum()
df.drop_duplicates(inplace=True)
df.info()

# %%
# Quiz
# # 读取./data/311-service-requests.csv，探索一下这个表格
# 看看那种抱怨最多

complaints = pd.read_csv('./data/311-service-requests.csv')


# %%
# Quiz
# 看看哪个区（borouhg）抱怨马路上噪声（Noise - Street/Sidewalk）的最多


# %%
# Quiz
# 选出所有Brooklyn区和噪音有关的抱怨，提示，可以使用`.str.contains("Noise")`


# %%

# 练习，画出covid-19 confiremed case变化趋势和增长率曲线，之画出确诊累计人数最多的5个国家

cases = pd.read_csv("covid-19-cases.csv")
# %%
