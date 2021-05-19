# %% [markdown]
'''
好这里我们给大家介绍一些额外的回归方法，主要用到sklearn。

这里给大家看一个预测房价的例子，就不画图了，主要通过rsquared来看性能

我们把线性模型作为baseline，然后我们来看SVM，KNeighbors和RandomForest

导入数据：
'''
# %% 
import pandas as pd

data = pd.read_csv('kc_house_data.csv').iloc[:, 2:]
data.dropna(inplace=True)
data.head()
# %%
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
y = data['price'].values
X = data.iloc[:, 1:].values

models = dict()
models['Linear'] = LinearRegression()
models['SVR'] = SVR()
models['KNR'] = KNeighborsRegressor()
models['RFR'] = RandomForestRegressor()
X_train, X_test, y_train, y_test = train_test_split(X, y)
print(len(y))
print(len(y_train))
print(len(y_test))
# %% [markdown]

'''
上面的train_test_split是用来分训练集和测试集的，
因为不涉及模型参数的迭代所以没有验证集。
'''
# %%

predictions = dict()
scores = dict()
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions[name] = model.predict(X_test)
    scores[name] = model.score(X_test, y_test)
    print('The R-squared of model {} is: {}'.format(name, scores[name]))

# %% [markdown]

'''
可以看到除了RandomForest之外其他的模型SVM和KNeighbors都表现得稀烂，
尤其是SVM，Rsquared竟然是负的。
可能是因为没有归一化，
归一化就是把你的数据映射到某一个范围内，一般是(0, 1)或者(-1, 1)，
我们下面要用的StandardScaler就是一种归一化方法，他是把数据缩放到均值为0，方差为1，
这样缩放的好处是他不改变原有的数据分布
'''
# %%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scale, X_test_scale = scaler.transform(X_train), scaler.transform(X_test)
for name, model in models.items():
    model.fit(X_train_scale, y_train)
    predictions['scaled_X{}'.format(name)] = model.predict(X_test_scale)
    scores['scaled_X{}'.format(name)] = model.score(X_test_scale, y_test)
    print('The R-squared of model scaled X {} is: {}'.format(name, scores['scaled_X{}'.format(name)]))
# %% [markdown]

'''
可以看到KNeighbors已经上来了，但SVM还是不行。
我们再归一化一下y：
'''
# %%
y_scaler = StandardScaler()
y_scaler.fit(y_train.reshape(-1, 1))
y_train_scale, y_test_scale = y_scaler.transform(y_train.reshape(-1, 1)).ravel(), y_scaler.transform(y_test.reshape(-1, 1)).ravel()
for name, model in models.items():
    model.fit(X_train_scale, y_train_scale)
    predictions['scaled_y{}'.format(name)] = model.predict(X_test_scale)
    scores['scaled_y{}'.format(name)] = model.score(X_test_scale, y_test_scale)
    print('The R-squared of model scaled y {} is: {}'.format(name, scores['scaled_y{}'.format(name)]))

# %% [markdown]

'''
## 作业1

用我们讲的房价的数据，到sklearn上找若干我们没讲过
的回归模型，自己玩一下，用train test split分割训练集
和验证集，给出完整的拟合、预测代码，并给出在验证集上的
r-squared.

可以参见 https://scikit-learn.org/stable/supervised_learning.html
'''

# %%
