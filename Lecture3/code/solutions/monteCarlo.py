

# %% [markdown]
'''
## Quiz
下面我们做个联系，假设小明每天早上去上班要做公交车，她早上出门有70%概率坐A路，30%概率坐B路，
A到公司有6站路，每站之间运行2或着3分钟，50%概率随机，然后每站停车0.5或着1分钟，50%概率随机
B到公司有8站路，每站之间运行1或着2分钟，50%概率随机，然后每站停车0.5或着1分钟，50%概率随机，
B车如果连续两站停留1分钟，下一段路一定是1分钟路程(从小明上车的这一站算起，并且小明起始站的
停车等待时间算入小明的上班花费时间内)。
求小平均多长时间到公司，如果小明8点上班，那么他每天几点出门坐车，才能保证90%都不迟到？

一下解答是错的！！，你猜猜错在哪？
'''

"""
求平均花费时间
"""
iteration = 10000                                                                          # 模拟出行次数，越多越精确
# 存放iteration次出行每次所花的时间
cost_times = []
for i in range(iteration):
    route = np.random.choice(["A", "B"], p=np.array(
        [0.7, 0.3]))                            # 每次出行的路线。线路A概率70%，线路B概率30%
    # 线路A有6站。即6次运行，6次停车等待
    if route == "A":
        # 依次存放6次运行的时间
        drive_cost_times = []
        # 依次存放6次停车等待的时间
        wait_cost_times = []
        for station in range(6):
            drive_cost_time = np.random.choice([2, 3], p=np.array(
                [0.5, 0.5]))              # 线路A每站行驶时间。运行2或着3分钟，50%概率随机
            wait_cost_time = np.random.choice([0.5, 1], p=np.array(
                [0.5, 0.5]))             # 线路A每站停车等待时间。停车0.5或着1分钟，50%概率随机
            drive_cost_times.append(drive_cost_time)
            wait_cost_times.append(wait_cost_time)
        cost_times.append(sum(drive_cost_times + wait_cost_times)
                          )                          # 计算这次出行所花的时间
    # 线路B有8站。即8次运行，8次停车等待
    else:
        drive_cost_times = []
        wait_cost_times = []
        for station in range(8):
            # 如果连续两站停留1分钟，下一段路一定是1分钟路程。触发此条件必须已经行驶2站以上
            if station < 2:
                drive_cost_time = np.random.choice(
                    [1, 2], p=np.array([0.5, 0.5]))
                wait_cost_time = np.random.choice(
                    [0.5, 1], p=np.array([0.5, 0.5]))
                drive_cost_times.append(drive_cost_time)
                wait_cost_times.append(wait_cost_time)
            else:
                # 如果连续两站停留1分钟，下一段路一定是1分钟路程。
                if (wait_cost_times[-1] == 1) & (wait_cost_times[-2] == 1):
                    drive_cost_time = 1
                    wait_cost_time = np.random.choice(
                        [0.5, 1], p=np.array([0.5, 0.5]))
                else:
                    drive_cost_time = np.random.choice(
                        [1, 2], p=np.array([0.5, 0.5]))
                    wait_cost_time = np.random.choice(
                        [0.5, 1], p=np.array([0.5, 0.5]))
                drive_cost_times.append(drive_cost_time)
                wait_cost_times.append(wait_cost_time)
        cost_times.append(sum(drive_cost_times + wait_cost_times))

print("平均花费时间：{}分钟".format(np.mean(cost_times))
      )                                    # 计算平均花费时间


"""
提前时间至少为多少，才能保证90%都不迟到
"""
cost_times_set = set(
    cost_times)                                                             # 以上iteration次模拟出行，到公司花费都时间就23种，在cost_times_set里由小到大无重复排列
cost_times = np.array(cost_times)
count = 0
for cost_time in cost_times_set:
    # 由小到大统计每个花费时间出现的次数，并累加到count
    cost_time_num = cost_times[cost_times == cost_time].size
    count = count + cost_time_num
    # 当count / iteration 》= 0.9时，对应的这个花费时间即是我们所求的最少提前时间
    if count / iteration >= 0.9:
        print("至少提前{}分钟出门，可保证至少90%都不迟到!".format(cost_time))
        break

# %%
