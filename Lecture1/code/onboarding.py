# %% [markdown]
"""
# Python 基础

大家好，在这里大家将学到python基本的语法，课一些关键的数据、运算操作。这些不仅仅是建模中有用，
在用python任何事情的时候都世纪初。Python不仅仅可以用来做机器学习，科学计算等等，在很多工程，科学应用上，
随手写一个python小程序都是很方便的，所以我们要熟练掌握这些基础。

这里你将学到：

1. 基本的数值操作变量和赋值
3. 流程控制
4. 定义函数
5. 基本数据结构
"""


# %% [markdown]
"""

## 基本的数值操作变量和赋值，这个你们肯定都会

在下面这个cell里面，首先：

1. 我们import了很多包，第一个就是numpy，可以看到我们用了`import`,这个就是导入一个包，我们就可以用这个包了，
如果我们后面加上`as xxx`那就是给这个包起了个别名，后面就可以用别名，少打几个字。虽然现代编辑器有了自动补全的功能，
但是我们还是会用别名，为什么呢？因为想什么`np，plt`这些，因为这些都是famous别名，大家都用，
这样你复制粘贴代码的时候就可以放心大胆的粘贴，老师就不会发现了，不对，应该主要是就能顺利运行了

"""

# %%


# 为什么要加个print，这是为了避免prettier自动格式化的时候把下面的cell和这个合并了
import numpy as np
from numpy import newaxis
import matplotlib.pyplot as plt
import numpy as np
import math
print('-----')

# %% [markdown]
'''
2. 这里定义了一个变量`a`，直接就可以定义，不需要声明类型，但是它是由类型，你可以看到他的类型是`int`。
这里要注意，在一个cell里面最后一行，如果不是赋值，这个一行表达式的之会被打印出来，不需要`print`,
但是，如果你要cell里面打印多个，就需要用print了.

3. 一个数字后面带上一个`.`就是浮点数

3. 可以看到，主要是除法，不管是不是证书类型都是float，没啥鸟用的冷知识

4. 后面就是把python当作计算器用、
'''
# %%
a = 1+2
print(type(a))

a = 1+2.
print(type(a))
# %%
a = 4/2
type(a)
# %%
5 * (4 + 3) - 2
# %%
2**4
# %%
9**0.5
# %%
11 % 4
# %%
11//4

# %% [markdown]
"""
## 流程控制

流程控制就是if else，循环那些，如果你学过别的语言就很明白，这里要注意个是：
python的程序block是通过缩进来定义的，连续同一个缩进级别的就是一个block，所以有有本书叫《Python从入门到买游标卡尺》。

![](2021-03-29-13-18-46)

下面看几个例子大家就明白了。
"""

# %%  [markdown]
"""
### If statement
"""

# %%
a = 0
if (a > 5):
    print("large")
elif (a > 0):
    print("mid")
else:
    print("samll")


# %% [markdown]

"""
### loop
"""
# %%
someList = [1, 2, 3]
for item in someList:
    # code block indented 4 spaces
    print(item)

# %%
n = 5
while n > 0:
    print(n)
    n = n - 1

# %% [markdown]

"""
### range function

有个比较好玩的东西叫做range，它可以产生一个iterable，让你来循环，里面的循环变量是一个范围，
按照一定步长来变化，可以看下面几个例子：
![](2021-03-29-13-24-21.png)
![](2021-03-29-13-25-09.png)
"""

# %%
for i in range(10):
    print(i)

# %%
for i in range(2, 5):
    print(i)

# %%
for i in range(1, 10, 3):
    print(i)

# %%
for i in range(-2, 2):
    print(i)

# %%
for i in range(3, -2, -2):
    print(i)

# %% [markdown]
'''
# Break Continue

这个和其它语言一样，我就不讲了。
'''

# %%
for i in range(10):
    if i % 2 == 0:
        continue
    if i > 6:
        break
    print(i)

# %% [markdown]
'''
# List Comprehension 

这是一个骚操作，大家可以看下面例子理解一下

就是可以写一个表达式，这个表达式里可以用循环变量
![](2021-03-29-13-26-32.png)
'''


# %%
squares = [d**2 for d in range(1, 11)]
print(squares)


# %% [markdown]
"""
## 使用和定义函数
"""

# %% [markdown]
'''
## 函数

当你有很多代码需要被重复利用的时候就可以定义一个函数。函数的参数，可以通过参数列表传进去。

函数里面可以用全局变量，只要前面定义过就行。
'''

# %%
# 调用一个函数
math.factorial(4)

# %%
# 定义一个函数试试
# 起个有意义的名字


def freeFall(g, t):
    h = 0.5*g*t**2
    return h


freeFall(9.8, 5)

# %%
# 函数是可以使用前面定义过的变量的
r = 2


def circleArea():
    return math.pi*r**2


circleArea()

# 当然这是不对的，应该把可能会改变的变量放到参数列表里面，如果你用面向对象的话，那再说我们这里不谈


# %% [markdown]
'''
## Quick Quiz
写一个函数生成一个斐波那契数列，输入时数列的个数，输出是这个数列的list
'''

# %%

# 写一个函数生成一个斐波那契数列


# %% [markdown]

"""
## 基本数据结构
这里讲一下你可能用的到的最基本的数据结构：  

1. 字符串
1. 字典
3. list
2. tuple

"""

# %%[markdown]
'''
## 字符串

字符串就是。。。字符串。在python里面万字符串是很简单的，你可以直接把他们加起来，就是连起来，
数字可以和字符串互转。
'''
# %%
a = "My dog's name is"
b = "Bingo"

c = a+" " + b
print(c)

print(b[1:3])
print(len(b))

d = 123
e = "4.56e5"
c = str(d) + e
print(c)

c = d+float(e)
print(c)


# %% [markdown]
"""
字典就像他的名字说的一样，你可以查找，你需要提供一key和一个value，你指定key的时候就可以读写这个key对应的value
![](2021-03-29-13-31-00.png)
"""

# %%
room = {"Emma": 309, "Jacob": 582, "Olivia": 764}

print(room["Emma"])

room["Fourier"] = 555

# %% [markdown]
'''
## 字典的Key可以是任何东西不一定是string，只要他是immutable的就行，你就认为是一般的单值的变量
'''
# %%
room[123] = 321

print(room[123])

# %% [markdown]
'''
## 你也可以创建一个空的字典然后往里面加东西
'''

# %%
d = {}
d["last name"] = "Alberts"
d["first name"] = "Marie"
d["birthday"] = "January 27"
print(d)
# %% [markdown]
'''
## 字典的keys or value是iterable的可以用来循环
'''

# %%

for i in d.keys():
    print(i+": "+d[i])


# %% [markdown]
'''
## List

List 就是数组，但是他是动态的，你除了可以修改它还可以往里面添加元素，
后面我们还会学numpy array，他们很像，但是区别也很大，要注意区分它们。
'''

# %%
# 创建一个list
a = [0, 1, 2, 3, 4, 5]
a.append(6)
print(a)


# %% [markdown]
'''
list 的元素可以是各种类型
'''
# %%
a.append("hello")
a.append(2+5j)
print(a)

# %% [markdown]
'''
1. 你可以index这个数组,负数数就是倒这来数

2. 也可以indexing 一个范围，他的规则是[start:end)

3. 也可以跳着[start:end:step]
'''

# %%

print(a[2])
print(a[-1])
print(a[-2])

# 也可以indexing 一个范围，他的规则是[start:end)
print(a[0:2])
# 也可以跳着
print(a[0:5:2])
print(a[2:])
print(a[2:-2])

# %% [markdown]
'''
list的+运算就是把两个list连起来。
'''
# %%
# list的+运算就是把两个list连起来。
b = a*2
print(b)
b = a+a

# 你可以这样网list里面插入
b = a[:5] + [101, 102] + a[5:]
print(b)

# 你也可以建造多维数组
a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print(a[:][0:-1])


# %%  [markdown]
'''
# Tuple
#  tuple其实就是不能变化的list，就是不能删除也不能增加里面的元素
'''
# %%
# tuple其实就是不能变化的list
a = (1, 2)
# 下面这一行会报错
a[0] = 0


# %% [markdown]
'''
## Quick Quiz

写一个函数，求组合，就是$c(n,k)=\frac{n!}{k!(n-k)!}$
'''

# %%
# 写一个函数，求组合

# %% [markdown]

"""
## Numpy Arry

刚刚上面讲的都是python核心的数据类型，而Numpy是一个python科学计算的包。它提供的numpy array是科学计算的核心，
可以把它理解成矩阵，实际上很不一样。numpy array和list有点像，但是也不一样，最大的不同是他只能存同样一个类型的变量，
他操作起来非常快，但是如果你要往里面添加元素的话，就会非常慢，因为在内存里面是连续存储的，你添加元素他就会销毁原来的
数据，然后再创建一个全新的。如果你经常添加删除元素，还是用list，在需要计算的时候在转换成numpy array

Numpy 还提供很多数学函数，numpy带的数学函数的特点是支持输入和输出为numpy array。

更多可以参考：https://numpy.org/doc/stable/user/quickstart.html
"""

# %% [markdown]

'''
## 创建nparray

1. 你可以把一个list转换为一个np array，但是要注意，这个list，必须每个元素都是同一个类型的，
如果不一样，它会被转换成一样的类型,那个可以兼容所有元素值的类型，所以最好采用一个类型。

2. 如果数组时多维的，最好每个维度都一样，这样会被转成一个多维的numpy array，
维度如果不一样的话，那个维度的数组会被当成一个元素。

3. 每个numpy array对象都有个shape属性，他表示这个数组的维度
'''

# %%
# 创建nparray
a = [1, 2, 3]
b = np.array(a)
print(type(b))

# %%
# 类型不一样
c = [1, True]
print(np.array(c))
c = [1, "1"]
print(np.array(c))

# %%
# 多维数组
c = np.array([[1, 2, 3], [1, 2, 3]])
print(c.shape)
print(c)

# 维度不一样
c = np.array([[1, 2, 3], [1, 2]])
print(c.shape)
print(c)
# %% [markdown]
'''
# 使用numpy自带的一些函数创建array
最常见的就是linspace，和range有点像，但他是支持小数，range不支持，他在起点和重点之间，均匀的产生，
n个数字
'''
b = np.linspace(1, 2, 11)
print(type(b))
print(b)

b = np.zeros((3, 3))
print(b)

b = np.linspace(1, 9, 9)
print(b.shape)
b = b.reshape(3, 3)
print(b.shape)
print(b)
# 可以将一个np array进行转置
print(b.T)

# %%
# shuffle 可以用来把一个array元素随机排列
b = np.linspace(1, 9, 9)
np.random.shuffle(b)
print(b)

# %% [markdown]
"""
###  indexing np array中的元素

这可以说时np array最重要的一个操作了，np array相比list，选择元素会更加方便灵活
"""


# %%
# reshape可以把数组变成别的形状，如果一个轴写-1代表他可以随意，因为其他的限定死了，这个也就是确定的，
# 可以省略，但是仅有一个可以是-1，
a = np.linspace(0, 24, 25).reshape(-1, 5)
print(a)

# 基本的slice操作
# 选择某一个维度全部数据，用":"

# 选择2列的所有行
print(a[:, 2])

# 选择1行的所有列
print(a[1, :])


# 选择1行的2到最后一列
print(a[1, 2:])

# 第0行，1，4列
print(a[0, [1, 4]])

# 每个两行，1，2列，每两列
# 起点:重点:每隔几个，不填就代表到第一个，或最后一个，或每隔一个
print(a[::2, 1:5:2])


# %%
a = np.random.randint(0, 30, 10)
print(a)
# 用一个list 来选择其中的元素
print(a[[2, 4, 6]])

# Masking，
# 生成一个list，这个list的元素代表原来array中每个元素是否满足这个条件
print(a % 3 == 0)
# 用一个bool数组来选择数组中的元素
print(a[a % 3 == 0])


# %%[markdown]
"""
## nparray的运算

一般np下面函数都是支持给一个值，或着给一个np array的，如果给多个一般相当于对每个进行运算
"""

# %%
b = np.linspace(1, 3, 3)
# 每个元素都相加
b = b+1
print(b)
b = b+b
print(b)
b = b*2
print(b)
b = b*b
print(b)
# 矩阵相乘
b = np.linspace(1, 3, 3)
b = np.matmul(b, b.T)
print(b)

# 一些支持的函数
b = np.linspace(0, 2*math.pi, 50)
c = np.sin(b)
plt.plot(b, c)
b = np.linspace(1, 3, 3)
print(sum(b))


# %% [markdown]

"""
### 练习
1. 写一个函数，一个球自由落体，输入时间t和时间间隔dt，计算t每间隔dt时间的位移。

2. 写代码生成下面这个些数据，同样颜色的为一个nparray，注意维度和方向：
   ![](2021-03-26-23-59-29.png)

3. 求下面这个积分  
 $\int_0^1\int_0^1\int_0^1(x^y-z)dxdydz$
"""


# %%
