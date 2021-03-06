# 什么是数学建模？

<!-- keywords:key1;key2; -->
<!-- description:this is a description -->
<!-- coverimage:![cover](cover.jpg) -->

## Pre

1. 这个课，虽然叫做电气工程建模与仿真，单并不完全是，我觉得叫做 intro mathematical modelling for engineering in python，Python 工程数学建模入门
2. 解释一下，首先是入门，这个课只有 32 学时，然后这个工程里面数学建模又是个非常庞大的 topic，所以只能入门，剩下的要靠大家在实际解决问题中自己去探索
3. 然后，这个是 mathematical modelling for engineering，所以数学的东西，不是我们的重点，重点是利用数学建模的方法来解决工程的问题，关键是解决工程上的问题。
4. 最后 python，后面我会将为什么要学 python，但是我们这个课中讲到的很多概念，你用别的工具一样也是可以做的

## 什么是模型？

在建模之前我们首先要明白什么是模型？

如果你查字典你会发现模型值得就是：“A model is an informative representation of an object, person or system.”

如果要翻译成中文，根据我多年的经验应该是：“模型是一个东西关键信息的一种表现。”

我们举几个例子：

1. 一个玩具车，他是一个模型，如果你没见过真正的车，你看过模型车，也就知道车子有四个轮子，几个门，然后在地上用轮子跑。
2. 一个地球仪也是一个模型，他告诉我们地球是圆的，哪里有陆地那里有海洋，这些信息，你如果不飞到太空去你是没法看到的。
3. 一个地图也是一个模型，世界地图，包含了整个地球但是他是平的，这并不是说地球是平的，我刚刚说所模型是一个东西关键信息的表现，对于一个地图，大的关键信息是方位和路线，引起他就没有表现地球的形状这个信息。
4. 一个图纸也是模型
5. 一个乐谱也是模型
   还有很多东西都是模型，比如 3D CGI 模型，模特姐姐等等，他们各有用途，信息的表现形式也不同。
   我怕们这里要学的数据建模的模型，是用数学的语言描述实际现实中的事物模型，比如：
   $F=ma$， 这个是个物理的模型，但是数学不一定只能用来描述物理，比如这个$\frac{dN}{dt}=rN(1-\frac{N}{K})$,这个是人口增长的模型。

**那模型有什么用呢**：

模型的作用是让你能够用你已经掌握的知道去理解、沟通、定义、量化、可视化或者模拟、预测真实世界里面某些事物。

理解：例如$F=ma$，你就可以知道，维持同样的加速度，如果质量越大需要的力就越大。

沟通：如果你向高速别人一个东西的原理，我相信一些数学公职，或者是流程图、结构图，电路图这种模型就很有用。

定义：有些东西是很虚无缥缈的难描述的，例如信息里里面很多概念，通过建模就可以描述，比如熵这个概念。或者是例如元素周期表，也是通过定义来描述元素的行为的。

量化：这个就很简单了，比如大家都知道，买彩票很难中奖，但到底多难呢？买多少多少张能中多少钱？这个通过概率模型、期望都能量化。

可视化：比如模特小姐姐穿上衣服、买家秀，或者一些系统的状态可以用波德图之类的表示，你一下就能看出来系统的大概的响应。

模拟：比如模拟一下如果 covid-19 pandemic 在不同的管控政策下的发展情况，

预测：路天气预报的模型

例如，玩具车，就能帮你可视化一个真的车，你可以把它拿起来翻来覆去的看，如果是真车估计就没那么方便了，地球仪也可以可视化地球。地图就可以量化一个城市的道路，或者从哪到哪的一条路线。图纸，或者乐谱就是定义用的。模特不仅可以可视化，也可以用于模拟，让你幻想一下自己穿上那件衣服的感觉。

数学建模也多半上面那些作用例如：$F=ma$，就能帮你理解，这个质量越大同样的加速度需要的力越大，这些也是量化分，废话这是个方正当然是量化的，也可以用来模拟，比如你想要发射一个火箭，知道火箭的推理和重量，你是可以模拟出他的加速度，速度，飞行轨迹的。

当然你这里要注意，前面说的“模型的作用是让你能够用你已经掌握的知道去理解、定义、量化、可视化或者模拟真实世界里面某些事物。”所以你必须已经掌握一些知识，这个就是所谓的模型的语言，对于数学建模就是基本的数学语言。

然后还要注意的是模型不是完美的，他只表达了真是事务的一些关键信息，忽略了很多别的信息的，因此模型是由偏差的。比如那个衣服穿在模型身上，仿真模拟出来的效果很好，逆转在身上就不一定了。数学模型一样有这个问题。

## 模型的特点

一切模型都有一下几个特点：

1. 抽象、简化，剥离掉不必要的细节和干扰因素，模型是用来解决问题的，你只用保留对解决问题关键的要素就可以了。例如
2. 都是形式化的（（formal）模型的一切都是有准去的定义和描述的，即便是随机和概率的模型，对概率分布也是有准确描述的
3. 一定会有一定的错误。模型不可能 100%的还原真实的情况，或多或少都是由误差的，这个误差也是可以定量分析的

## 什么是建模

建模就是建立模型的一个操作，废话。

那怎么建模的？一般有几种方法：

1. scale，就是缩小或方法，在小东西上得到的结论也许可以放大，对大东西也灵，或者反过来。
2. Symbolize：就是符号化，对关键信息进行抽象，发明对应的符号
3. Quantify：就是量化，这里可以理解为列方程
4. Simplify：简化，就是把没用的东西去掉，剩下的就是一个模型了

以上方法对于数学建模同样适用，而且大家记住，很多时候这个是混合起来使用的，并不是只用一种方法。

## 什么是数学建模

数学建模官方的解释是，MODELING IS A PROCESS THAT USES MATH TO REPRESENT,ANALYZE, MAKE PREDICTIONS,OR OTHERWISE PROVIDE INSIGHT INTO REAL-WORLD PHENOMENA，其实和之前说的一样

其实刚刚讲了这么多建模，数学建模就包含在上面了，数学建模就是建模和其他的建模没啥区别，最大的特点就是他用的是数学的语言来描述的。

这就是我们这个课程要学的，我要是这么快就讲完了还上什么课。所以大家慢慢学什么是数学建模。

## 数学建模的步骤

1. **首先要定义问题**，这一步是最重要的，他把一个工程问题，变成一个数学课描述的问题，比如问这个那个过山车最刺激，那首先你就要定义什么叫最刺激，是速度最快，还是加速度最大，呕吐人数最多，还是大家的评价，还是你根据上面几个指标综合出来一个评级指标？
2. **然后需要定义一些参数和变量**，就是根据上一步的问题定义，把相关的量都用变量或者是参数表示出来，确定参数和变量的时候需要根据你对问题的定义来做一些假设，比如你决定用加速度来评价刺激程度，那用户的评价这个数据就可以扔掉了不需要分析，而过山车的颜色，也不重要，但是他的速度，落差，轨道轨迹就很重要，关于什么是变量什么是参数这个我们后面马上具体说
3. **然后就是建模求解**，根据你对问题的定义和你选区的变量与参数，从你的数学工具箱中，找到一个建模工具，比如你要用加速度评价过山车，那你多半需要使用微分方程，如果使用评价，多半就要用到统计，等等。选择好工具之后，就是建模并且算出来你要的东西
4. **然后就是评价你的模型**，模型都是有偏差的，你一定要有一个手段评价你模型数据的正确成都，有时候你会建立多个模型，你需要从多个模型中选出一个你觉得对解决问题最有价值的。有时候由于建模的时候加入过过的的假设，或者是简化的过度了，导致模型出来的结果不不合理，比如你算出来要终结 covid-19 pandemic 的方法是引发第三次世家大战，这显然就不合理。

## 一些重要的概念

数学建模里面模型一般可以用一个函数表示，这个函数是广义的函数哈，我们抽象的称他为$y=f(x,\theta)$

可以看到这个模型里面除了这个函数，还有几个字母。$x，y，\theta$，这些是啥？

### Parameters 参数 & Variables 变量

模型里面有参数和变量这两个东西，参数一般用$\theta$表示，变量一般就是$x，y$表示。它们有什么区别呢？

**参数 Parameter 的定义是**：Parameters are usually used for defining some characteristics of the modeled object.

他的意思是说参数，是定义了一个模型的特征。例如一个车的模型，他的参数可以有，车的马力，车的重量，甚至有几个轮子。对于一个车来说这些量是固定的，不会改变，如果改变和几个量就变成了不同的模型。比如一个法拉利可以有 500 马力，而一辆特斯拉可以有 1100 马力。

**变量的定义是**：Variables are usually describing the state of the model.

就是说变量描述的是这个模型的状态，比如输入，输出，系统内部状态这些。例如一辆 500 匹的法拉利，你油门踩 80%，5s 之后，速度是 100km/h。之前俺说了 500 匹马力是参数，是不变的。而油门踩 80%是一个输入，5s 之后是时间，速度 100km/h 这都是变量。

这里要说一下变量有两种，一种叫 dependent variable，一种叫 independent variable。independent variable，就是你可以去不同的值，而 dependent variable 的值就是依赖 independent variable 的。这个不是绝对的。例如上面的例子，你可以认为速度是 dependent variable，他一来你踩多深的油门，和过了多久。但是如果你想知道的是踩多深的油门才能再 5s 的时候达到 100km？那么才多深的油门就变成了 dependent variable 了。

下面大家自己试试看：$H=\frac{1}{2}gt^2$，这个是自由落体的位移公式也是个模型，大家看看什么是变量什么是变量什么是参数？
