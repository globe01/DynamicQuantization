import torch
import numpy as np


# 单峰正态分布
def generate_single_peak_data():
    np.random.seed(42)
    data = np.random.normal(loc=0, scale=1.5, size=10000)
    return torch.tensor(data)

# 双峰分布（两个远离的正态分布）
def generate_double_peak_data():
    np.random.seed(42)
    data = np.concatenate([
        np.random.normal(loc=-10, scale=1.0, size=5000),
        np.random.normal(loc=10, scale=1.0, size=5000)
    ])
    return torch.tensor(data)


# 均匀分布
def generate_uniform_data():
    np.random.seed(42)
    data = np.random.uniform(low=-10, high=10, size=10000)
    return torch.tensor(data)

# 指数分布
def generate_exponential_data():
    np.random.seed(42)
    data = np.random.exponential(scale=2.0, size=10000)
    return torch.tensor(data)

# 多峰分布
def generate_multi_peak_data():
    np.random.seed(42)
    data = np.concatenate([
        np.random.normal(loc=-10, scale=0.5, size=1500),
        np.random.normal(loc=-5, scale=1.0, size=2000),
        np.random.normal(loc=0, scale=1.5, size=3000),
        np.random.normal(loc=5, scale=1.0, size=2500),
        np.random.normal(loc=10, scale=0.5, size=1000)
    ])
    return torch.tensor(data)

# 三角分布，更适合用于测试非均匀量化
def generate_triangular_data():
    np.random.seed(42)
    data = np.random.triangular(left=-10, mode=0, right=10, size=10000)
    return torch.tensor(data)

# 泊松分布（非对称离散分布）
def generate_poisson_data():
    np.random.seed(42)
    data = np.random.poisson(lam=5, size=10000)
    return torch.tensor(data, dtype=torch.float32)

# 极端值分布（带有离群点的正态分布）
def generate_extreme_values_data():
    np.random.seed(42)
    data = np.concatenate([
        np.random.normal(loc=0, scale=1.0, size=9800),
        np.random.normal(loc=0, scale=20.0, size=200)  # 离群点
    ])
    return torch.tensor(data)


# 高斯分布+均匀噪声
def generate_gaussian_with_uniform_noise():
    np.random.seed(42)
    data = np.random.normal(loc=0, scale=1.5, size=8000) + np.random.uniform(low=-1, high=1, size=8000)
    return torch.tensor(data)
