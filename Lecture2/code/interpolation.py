
# %% [markdown]
'''
 # 电气工程建模与仿真——Interpolation
 
 插值是一种给定若干个已知的离散数据点，在一定范围内推求新的数据点的过程或方法。
 
 这里我们主要介绍`scipy.interpolate.interp1d`类中的若干插值方法。
 
 调包：
'''

# %%
import numpy as np
from scipy.interpolate import interp1d  # 官方文档：https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html
from matplotlib import pyplot as plt
print('---')

# %%
x = np.linspace(0, 2*np.pi, 10)  # 初始已知10个点
y = np.sin(x)
x_new = np.linspace(0, 2*np.pi, 50)  # 目标插值50个点
plt.plot(x, y, '.')


# %%
kinds = ['nearest', 'next', 'linear', 'cubic']
markers = ['v', '^', '<', '>']
plt.figure(figsize=(20, 10))
plt.scatter(x, y, marker='*', s=800, label='raw')

for i in range(len(kinds)):
    f = interp1d(x, y, kind=kinds[i])
    y_new = f(x_new)
    plt.scatter(x_new, y_new, marker=markers[i], label=kinds[i])
plt.legend(loc='best')

# %% [markdown]
 '''
 `interp1d`类中提供了若干种插值方法，包括"linear", "nearest", "nearest-up", "zero", "slinear", "quadratic", "cubic", "previous"以及"next"。
 
 其中：
 
 "zero", "slinear", "quadratic", "cubic"分别代表零次、一次、二次、三次样条插值
 
 "previous", "next"分别代表取当前点的上一个或下一个点
 
 "nearest", "nearest-up"取最近的点，二者区别在于，插值半整数时(如0.5)，"nearest"向下取整，"nearest-up"向上取整
 
 "linear"即把相邻点相连并均等取值
 
 要注意插值与后面的拟合(fitting)存在不同
 
 插值必过给定的已知离散点，拟合时可以不过点，因为要满足最佳逼近
 
 对底层方法感兴趣的同学可以参考《数值分析》
 
 `scipy.interpolate`这一sub-package中提供了其他多种应用于不同场合的插值方法，感兴趣的同学可以自行查阅[官方文档](https://docs.scipy.org/doc/scipy/reference/interpolate.html)
'''
