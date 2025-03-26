from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpMaximize, LpStatus

# 课程数据
credit = [5, 4, 4, 3, 4, 3, 2, 2, 3]  # 每门课的学分
math_courses = [1, 2, 3, 4,5]  # 数学课程编号
or_courses = [3, 5, 6, 8, 9]  # 运筹学课程编号
cs_courses = [4, 6, 7, 9]  # 计算机课程编号

# 定义问题（最少课程）
prob = LpProblem("Course Selection", LpMinimize)

# 定义变量
x = [LpVariable(f"x{i+1}", cat="Binary") for i in range(9)]

# 目标函数 1：最少选课
prob += lpSum(x)

# 约束：至少选 2 门数学课
prob += lpSum(x[i-1] for i in math_courses) >= 2
# 约束：至少选 3 门运筹学课
prob += lpSum(x[i-1] for i in or_courses) >= 3
# 约束：至少选 2 门计算机课
prob += lpSum(x[i-1] for i in cs_courses) >= 2

# 先修课约束
prob += x[2] <= x[0]  # 最优化方法 需要 微积分
prob += x[2] <= x[1]  # 最优化方法 需要 线性代数
prob += x[4] <= x[0]  # 应用统计 需要 微积分
prob += x[4] <= x[1]  # 应用统计 需要 线性代数
prob += x[3] <= x[6]  # 数据结构 需要 计算机编程
prob += x[5] <= x[6]  # 计算机模拟 需要 计算机编程
prob += x[7] <= x[4]  # 预测理论 需要 应用统计
prob += x[8] <= x[0]  # 数学实验 需要 微积分
prob += x[8] <= x[1]  # 数学实验 需要 线性代数

# 求解
prob.solve()
print("选课方案：")
for i in range(9):
    if x[i].varValue == 1:
        print(f"选修课程 {i+1}")

print(f"最少课程数: {lpSum(x[i].varValue for i in range(9))}")

# 目标函数 2：最大学分
prob.objective = lpSum(credit[i] * x[i] for i in range(9))
prob.solve()
print(f"最大学分方案的总学分: {lpSum(credit[i] * x[i].varValue for i in range(9))}")
