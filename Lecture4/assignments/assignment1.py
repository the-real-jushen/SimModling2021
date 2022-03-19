# %% [markdown]
# 请把下面的问题线性化以后求解：

#![](../code/pics/pic14.png)
# $$
# S=\left[
# \begin{matrix}
# 0 & -7/12 & -1/4 \\
# 0 & -5/12 & -3/4 \\
# 0 & 5/12 & -1/4
# \end{matrix}
# \right]
# $$
#
# 目标函数
# $$
# min:
# \begin{cases}
# F &= 0.0007P_{G1}^2+0.30P_{G1}+4 \\
#   &+ 0.0004P_{G2}^2+0.32P_{G2}+3
# \end{cases}
# $$
# 运行约束：
# 1. 节点功率平衡方程
# $$ P_{G1} + P_{G2} = 100 $$
# 2. 发电机出力上下限
# $$ 30 \le P_{G1} \le 150 $$
# $$ 0 \le P_{G2} \le 50 $$
# 3. 线路潮流上下限
# $$
# -\left[
# \begin{matrix}
# 30 \\
# 80 \\
# 50 \\
# \end{matrix}
# \right]
# \le S
# \left[
# \begin{matrix}
# P_{G1} \\
# P_{G2} \\
# -100 \\
# \end{matrix}
# \right]
# \le
# \left[
# \begin{matrix}
# 30 \\
# 80 \\
# 50 \\
# \end{matrix}
# \right]
# $$

# %%
