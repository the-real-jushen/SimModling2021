'''
### 练习
1. 写一个函数，一个球自由落体，输入时间t和时间间隔dt，计算t每间隔dt时间的位移。
'''
# %%
# %%
import math
from scipy import optimize
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt


def freeFall(t, dt):
    time = np.arange(0, t, dt)
    return 0.5*9.8*time**2


result = freeFall(10, 1)
result

# %%[markdown]

'''
2. 写代码生成下面这个些数据，同样颜色的为一个nparray，注意维度和方向：
   ![](2021-03-26-23-59-29.png)

'''
# %% [markdown]
'''
3. 求下面这个积分  
 $\int_0^1\int_0^1\int_0^1(x^y-z)dxdydz$
'''
# %%


def f(x, y, z):
    return x**y-z


xn = np.linspace(0, 1, 100)
grid = np.meshgrid(xn, xn, xn)

allPoints = f(*grid)

I = np.average(allPoints)
I

# %% [markdown]
'''
## 练习 4
1. 在一个 3 维空间内，物质的密度分布为$\rho=x^2y^2z^2$,求一个圆锥体，底面为 xy 平面，圆心在原点，
半径为 1，高为 1 的圆锥体的质量
'''
# %%


def f(x, y, z): return x**2*y**2*z**2


I = integrate.tplquad(f, -1, 1, lambda x: -1*(1-x**2)**0.5, lambda x: (1-x**2)
                      ** 0.5, lambda x, y: 0, lambda x, y: (1-(x**2+y**2)**0.5))
I


# %%
