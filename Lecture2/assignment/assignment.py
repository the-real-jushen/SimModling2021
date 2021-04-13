# %% [markdown]

'''
## 作业1

画出covid-19 confiremed case变化趋势和增长率曲线，只画出确诊累计人数最多的5个国家
数据课：`cases = pd.read_csv("covid-19-cases.csv")`读取

## 作业2

"data.csv"中存有幅值220V，频率50Hz，初相位为0，采样频率10kHz，时间长度为1s的电压数据。
已知电压中存在幅值、相位各不相同的谐波(谐波幅值低于基波1%可忽略不计)，尝试：
1. 将给定数据降采样至1kHz；
2. 对降采样后的数据进行FFT分析，确定存在何种频率的谐波(幅值很小的（例如小于1%）谐波可以忽略)，
及其幅值、相位，计算总谐波失真（THD）；
3. 滤除谐波，并将滤波后的信号还原，分别与目标信号(y=220*sin(2*pi*50*x))与原始信号比较。

'''
# %%