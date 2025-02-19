# %% [markdown]

'''
Quiz

>exp4 这个是年龄，血压，是否吸烟和未来10年发生中风的概率的关系，现在让你预测一个人是否未来会发生中风。

| Age x1 | Blood Pressure x2 | Smoker x3 | % Risk of Stroke   over Next 10 Years Y |
|--------|-------------------|-----------|-----------------------------------------|
| 63     | 129               | No        | 7                                       |
| 75     | 99                | No        | 15                                      |
| 80     | 121               | No        | 31                                      |
| 82     | 125               | No        | 17                                      |
| 60     | 134               | No        | 14                                      |
| 79     | 205               | Yes       | 48                                      |
| 79     | 120               | Yes       | 36                                      |
'''
# %%
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_excel('exp4.xlsx')
df.replace(to_replace=('Yes', 'No'), value=(1, 0), inplace=True)
df.head()

# %% [markdown]
'''
分别写个简单的线性模型和多项式模型吧：
'''
# %%
# 线性模型

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values.reshape(-1, 1)

X_int = sm.add_constant(X)

model = sm.OLS(y, X_int)
res = model.fit()
res.summary()

# %%
# 多项式模型

poly_reg = PolynomialFeatures(degree=3)
poly_X = poly_reg.fit_transform(X)
poly_X_int = sm.add_constant(poly_X)
poly_model = sm.OLS(y, poly_X_int)
poly_res = poly_model.fit()
poly_res.summary()
# %%
