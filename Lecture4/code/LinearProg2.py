# %% [markdown]
# # 电力系统经济运行问题的数学建模与仿真与非线性优化
# ![avatar](./pics/pic6.png)
# ![avatar](./pics/pic7.png)

# %% [markdown]
# # 电力系统最优潮流的定义
# ## 定义
# 当系统的结构参数和负荷情况都已**给定**时，调节可利用的**控制变量**（如发电机输出功率、可调变压器抽头等）来找到能满足所有
# **运行约束条件**的，并使系统的某一**性能指标**（如发电成本或网络损耗）达到**最优值**下的潮流分布
# ![avatar](./pics/pic8.png)
#
# **控制变量**：发电机输出功率、可调变压器抽头等
#
# **运行约束**：功率平衡、发电机出力、线路容量、节点电压等
#
# **性能指标**：发电成本或网络损耗等

# %% [markdown]
# # 电力系统最优潮流数学模型
# ## 基于交流潮流的电力系统最优潮流模型
# ### 目标函数
# $$ min f(x)=\sigma_{i \in S_G} [a_iP_{Gi}^2+b_iP_{Gi}+c_i] $$
# ### 运行约束
# 1. 节点功率平衡方程
# $$ V_i\sum_{j=1}^nV_j(G_{ij}cos\theta_{ij} + B_{ij}sin\theta_{ij})-P_{Gi}+P_{Di}=0\quad \forall i=1, \cdots, N $$
# $$ V_i\sum_{j=1}^nV_j(G_{ij}sin\theta_{ij}-B_{ij}cos\theta_{ij})-Q_{Gi}+Q_{Di}=0 \quad \forall i = 1, \cdots, N $$
# 2. 各变量上下限
# $$ \underline{P}_{Gi} \le P_{Gi} \le \overline{P}_{Gi} \quad \underline{Q}_{Gi} \le Q_{Gi} \le \overline{Q}_{Gi} \quad \underline{V}_{i} \le V_{i} \le \overline{V}_{i} $$
# $$ \underline{P}_{ij} \le P_{ij} = V_iV_j(G_{ij}cos\theta_{ij} + B_{ij}sin\theta_{ij}) - V_i^2G_{ij} \le \overline{P}_{ij} $$
# %% [markdown]
# ### 目标函数
# $$ min f(x)=\sigma_{i \in S_G} [a_iP_{Gi}^2+b_iP_{Gi}+c_i] $$
# ### 运行约束
# 1. 节点功率平衡方程
# $$ \sum_{j=1}^n\frac{\theta_{ij}}{X_{ij}} - P_{Gi}+P_{Di} = 0 \quad \Rightarrow \quad P_N=Y_\theta \quad \Rightarrow \quad \sum_{i\in S_G}P_{Gi}=D $$
# 2. 各变量上下限
# $$ \underline{P}_{Gi} \le P_{Gi} \le \overline{P}_{Gi} \quad \Rightarrow \quad \underline{P}_{G} \le P_{G} \le \overline{P}_{G} $$
# $$ \underline{P}_{ij} \le \frac{\theta_{ij}}{X_{ij}} \le \overline{P}_{ij} \quad \Rightarrow \quad \underline{P}_{L} \le SP_{N} \le \overline{P}_{L} $$
# %% [markdown]
# ### 目标函数
# $$ min \sum_{k\in K}\sum_{j\in J}[c^p(j, k) + c^u(j, k) + c^d(j, k)] $$
# ### 功率平衡约束
# $$ \sum_{j\in J}p(j, k) = D(k) $$
# ### 机组最大最小出力约束
# $$ \underline{p}(j, k) \le p(j, k) \le \overline{p}(j, k) $$
# $$ \underline{P}_jv(j, k) \le \underline{p}(j, k) $$
# $$ \overline{p}(j, k) \le \overline{P}_jv(j, k) $$
# ### 系统正、负旋转备用约束
# $$ \sum_{j\in J}\overline{p}(j, k) \ge D(k) \cdot (1+L\%) $$
# $$ \sum_{j\in J}\underline{p}(j, k) \le D(k) $$
# ### 线路容量约束
# $$ -f_{lim} \le Sp \le f_{lim} $$
# ### 常规机组爬坡容量约束
# $$
# \begin{aligned}
# \overline{p}(j, k)
# &\le p(j, k - 1) + RU_jv(j, k - 1) \\
# &\le SU_j[v(J, k)-v(j, k - 1)] \\
# &\le \overline{P}_j[1-v(j, k)]
# \end{aligned}
# $$
# $$ \overline{p}(j, k) \le \overline{P}_jv(j, k + 1) + SD_j[v(j, k) - v(j, k + 1)] $$
# $$
# \begin{aligned}
# p(j, k - 1) - \underline{p}(j, k)
# &\le RD_jv(j, k) \\
# &+ SD_j[v(j, k - 1) - v(j, k)] \\
# &+ \overline{P}_j[1-v(j, k - 1)]
# \end{aligned}
# $$
# ### 最小启停时间约束
# $$ \sum_{k=1}^{G_i}[1-v(j, k)] = 0 $$
# $$ \sum_{n=k}^{k+UT_j-1}v(j, n) \ge UT_j[v(j, k)-v(j, k - 1)] $$
# $$ \sum_{n=k}^{T} \{v(j, n)- [v(j, k) - v(j, k - 1)] \} \ge 0$$
# $$ \sum_{k=1}^{L_i}v(j, k) = 0 $$
# $$ \sum_{n=k}^{k+DT_j-1} [1 - v(j, n)] \ge DT_j[v(j, k - 1) - v(j, k)] $$
# $$ \sum_{n=k}^{T} \{ 1-v(j, n) - [v(j, k - 1) - v(j, k)] \} \ge 0 $$
# %%
# %% [markdown]
# ## 如何建模和求解下面这个问题？
# <div align=center><img src='./pics/pic10.png' /></div>
#
# **数学模型：**
# $$ min: \{ F=F_1(P_{G1})+F_2(P_{G2}) \} $$
# $$ st.: P_{G1}+P_{G2}=P_{LD} $$
# **参数：**
# $$ F_1=0.0007P_{G1}^2+0.30P_{G1}+4 $$
# $$ F_2=0.0004P_{G2}^2+0.32P_{G2}+3 $$
# $$ P_{LD2}=100MW $$
# **求解：**
# $$
# min:
# \begin{cases}
# F &= 0.0007P_{G1}^2+0.30P_{G1}+4 \\
#   &+ 0.0004P_{G2}^2+0.32P_{G2}+3
# \end{cases}
# $$
# $$ st.: P_{G1}+P_{G2}=100 $$

# %%[markdown]
'''

 ## 将这个问题线性化
 上面的问题是一个非线性问题。由于机组出力有最大最小的限制，连接二次曲线首末两端可以对目标函数进行线性化处理得到下式：

![](pics/pic11.png)

 $$
 min:
 \begin{cases}
 F = 0.72P_{G1}-52 \\
   +0.44P_{G2}-5
 \end{cases}
 $$
 $$
 \begin{aligned}
 st.:  & P_{G1}+P_{G2}=100 \\
 & 30 \le P_{G1} \le 150 \\
 & 0 \le P_{G2} \le 50
 \end{aligned}
 $$
'''

# %%
