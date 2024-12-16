import torch
import numpy as np
import time

# 设置动态随机种子
def set_dynamic_seed():
    seed = int(time.time() * 1000000) % (2**32 - 1)  # 保证种子是合法的 32 位整数
    np.random.seed(seed)
    return seed

# 单峰正态分布
def generate_single_peak_data():
    set_dynamic_seed()  # 每次调用生成不同的随机数
    data = np.random.normal(loc=0, scale=1.5, size=10000)
    return torch.tensor(data)

# 双峰分布
def generate_double_peak_data():
    set_dynamic_seed()
    data = np.concatenate([
        np.random.normal(loc=-10, scale=1.0, size=5000),
        np.random.normal(loc=10, scale=1.0, size=5000)
    ])
    return torch.tensor(data)

# 均匀分布
def generate_uniform_data():
    set_dynamic_seed()
    data = np.random.uniform(low=-10, high=10, size=10000)
    return torch.tensor(data)

# 指数分布
def generate_exponential_data():
    set_dynamic_seed()
    data = np.random.exponential(scale=2.0, size=10000)
    return torch.tensor(data)

# 多峰分布
def generate_multi_peak_data():
    set_dynamic_seed()
    data = np.concatenate([
        np.random.normal(loc=-10, scale=0.5, size=1500),
        np.random.normal(loc=-5, scale=1.0, size=2000),
        np.random.normal(loc=0, scale=1.5, size=3000),
        np.random.normal(loc=5, scale=1.0, size=2500),
        np.random.normal(loc=10, scale=0.5, size=1000)
    ])
    return torch.tensor(data)

# 三角分布
def generate_triangular_data():
    set_dynamic_seed()
    data = np.random.triangular(left=-10, mode=0, right=10, size=10000)
    return torch.tensor(data)

# 泊松分布
def generate_poisson_data():
    set_dynamic_seed()
    data = np.random.poisson(lam=5, size=10000)
    return torch.tensor(data, dtype=torch.float32)

# 极端值分布
def generate_extreme_values_data():
    set_dynamic_seed()
    data = np.concatenate([
        np.random.normal(loc=0, scale=1.0, size=9800),
        np.random.normal(loc=0, scale=20.0, size=200)  # 离群点
    ])
    return torch.tensor(data)

# 高斯分布 + 均匀噪声
def generate_gaussian_with_uniform_noise():
    set_dynamic_seed()
    data = np.random.normal(loc=0, scale=1.5, size=8000) + np.random.uniform(low=-1, high=1, size=8000)
    return torch.tensor(data)
