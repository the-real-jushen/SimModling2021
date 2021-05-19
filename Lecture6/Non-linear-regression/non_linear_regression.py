# %% [markdown]
## 非线性回归

'''



#  数据科学介绍1.x：非线性回归

这里我们课上不讲，大家感兴趣自己回去看，可简单了

之前讲了线性回归，再讲一下非线性回归。当你建立的模型有一个根据规则建立的公式的时候，如果这个公式不满足线性回归的时候就需要使用非线性回归了。

如下面这个例子：

>exp 5在化学动力学反应过程中，建立了一个反应速度和反应物含量的数学模型，形式为：

$$
y=\frac{\beta_4x_2-\cfrac{x_3}{\beta_5}}{1+\beta_1x_1+\beta_2x_2+\beta_3x_3}
$$

现在测到一组参考数据，求模型参数 $\beta_1..\beta_5$

| x1 氢 | x2 戊烷 | x3 异构戊烷 | y 反应速率    |
|-----|-----|-----|-------|
| 470 | 300 | 10  | 8.55  |
| 285 | 80  | 10  | 3.79  |
| 470 | 300 | 120 | 4.82  |
| 470 | 80  | 120 | 0.02  |
| 470 | 80  | 10  | 2.75  |
| 100 | 190 | 10  | 14.39 |
| 100 | 80  | 65  | 2.54  |
| 470 | 190 | 65  | 4.35  |
| 100 | 300 | 54  | 13    |
| 100 | 300 | 120 | 8.5   |
| 100 | 80  | 120 | 0.05  |
| 285 | 300 | 10  | 11.32 |
| 285 | 190 | 120 | 3.13  |

我们需要用一个函数和前面的很象，叫 `fitnlm` fit non-linear-model。
代码如下：

```matlab
%%define your model equation using a anonymous function
f=@(B,X) (B(4)*X(:,2)-X(:,3)/B(5))./(1+B(1)*X(:,1)+B(2)*X(:,2)+B(3)*X(:,3))

%% feed the data and the function to the fit
%this is the initial value
B0=[0.1 0.1 0.1 0.1 0.1]
model=fitnlm(exp5,f,B0)

%% make prediction
y=model.predict([470 300 10])
```

可以看到我们用了 `@(B,X) ...` 这个匿名函数的语法定义我们的一个模型公式，B时参数，X时independent variable，都是向量。这里要注意的时，后面fit的时候是吧整个样板矩阵全部送进去的，这里这里要去对应的一列，比如X(:,2)就是把X矩阵的第二列所有行取出来，这个函数的返回值就是我们样本的所有的y，就是数据表里面Y那一列。

然后fit的时候需要给一个参数的初值。因为非线性回归不想线性回归一样有现成公式可以借，线性回归的损失函数是一个凸函数可以直接算，这个非线性回归的损失函数不知道是个啥，只能通过如梯度下降之类的优化算法，所以需要一个迭代的初值，这个就不多讲了。

这个fit方法输入的数据是一个table，我们导入的数据，它自动把最后一列作为dependent variable了，我们对于赢得处理数据就行。当让你可以用矩阵的语法把这个传进去 `fitnlm(X,Y,f,B0)` 这种。

当你得到模型以后可以看到输出是：

'''
# %%
import pandas as pd
from scipy.optimize import curve_fit

df = pd.read_excel('exp5.xlsx')
df.head()


# %%
def objective(X, a, b, c, d, e):
    return (d * X[1] - X[2] / e) /  (1 + a * X[0] + b * X[1] + c * X[2])
"""
这里我测过，objective的参数矩阵不能封装成列表；传进curve_fit()的xdata和ydata不能写成X和y然后放进去(X和y如下面定义)
后者可以参考 https://stackoverflow.com/questions/20769340/fitting-multivariate-curve-fit-in-python
"""
# X = df.iloc[:, :-1].values.reshape(3, -1)
# y = df.iloc[:, -1].values.reshape(1, -1)
data = df.values.transpose()
popt, _ = curve_fit(objective, data, data[-1, :])
popt

# %% 
from sklearn.metrics import r2_score
X = data[:-1]
y = data[-1]
a, b, c, d, e = popt
pred_y = (d * X[1] - X[2] / e) /  (1 + a * X[0] + b * X[1] + c * X[2])

r2 = r2_score(y, pred_y)

adj_r2 = 1 - (1 - r2) * (len(y) - 1) / (len(y) - X.shape[0] - 1)

print(r2)

print(adj_r2)
# %% [markdown]

'''
可以看到这个模型的rsquare非常接近于1。
为什么呢？（虽然有可能这个题的数据是来根据模型生成的）
在有些情况下，你可以通过某些物理和化学规律建立一个模型的公式，
这个时候通过数据找到一组符合这个规律和实验数据的模型肯定是最准的。
所以等你建立模型的时候，可以根据已知的规律建立模型公式的话，
一定比用一个多项式这种通用的模型效果好。但是这种情况非常少。
'''