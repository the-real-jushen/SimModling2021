# %% [markdown]

'''
我们可以尝试一个例子，不过这个例子还是给你们自己玩，我只给一些提示，

>exp 6.1 泰坦尼克这个船沉了，有很多人淹死了有很多人没有。其实你可以根据他们的一些情况判断他们活下来的概率。
'''

# %%
import pandas as pd

df_train = pd.read_csv('exp-6.1-train.csv')
df_train.head()
# %%
def pre_processing_titanic(df):
    df = df[['Pclass', 'Sex', 'Age','SibSp','Parch','Fare','Embarked', 'Survived']]
    df = df.dropna()
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'])
    return df.loc[:, df.columns != 'Survived'].values, df.loc[:, df.columns == 'Survived'].values
# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
X, y = pre_processing_titanic(df_train)
train_X, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)
log_model = LogisticRegression().fit(train_X, train_y)

# %%
pred_y = log_model.predict(train_X)
# %%
from sklearn.metrics import plot_confusion_matrix, confusion_matrix
plot_confusion_matrix(log_model, test_x, test_y)