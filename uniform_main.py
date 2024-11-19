# 全使用均匀区间量化


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import torch

from quantization_utils import (
    n1, n2, b1, b2, half_range_b2, half_range, generate_data, interval_based_quantize, interval_based_quantize_dequantize,
    interval_based_quantize_eachprint, interval_based_quantize_map, 
    dynamic_quantize_level_reduce, dynamic_quantize_level_reduce_dequantize, map_to_target_range
)

from distribution import (
    generate_single_peak_data, generate_double_peak_data,
    generate_uniform_data, generate_exponential_data,
    generate_multi_peak_data, generate_triangular_data,
    generate_poisson_data, generate_extreme_values_data,
    generate_gaussian_with_uniform_noise
)



# 主程序
data = generate_data()  # 生成数据

# data = generate_single_peak_data()  # 生成单峰均匀分布数据
# data = generate_double_peak_data()  # 生成双峰均匀分布数据
# data = generate_uniform_data()  # 生成均匀分布数据
# data = generate_exponential_data()  # 生成指数分布数据
# data = generate_multi_peak_data()  # 生成多峰分布数据
# data = generate_triangular_data()  # 生成三角分布数据
# data = generate_poisson_data()  # 生成泊松分布数据
# data = generate_extreme_values_data()  # 生成极端值分布数据
# data = generate_gaussian_with_uniform_noise()  # 生成高斯分布+均匀噪声数据




# 画图
plt.figure(figsize=(15, 5))  # 画布总的大小
# 先画原始数据分布图
plt.subplot(1, 3, 1)
plt.hist(data.numpy(), bins=100, color='blue', alpha=0.7)
plt.title("Original Data Distribution")
plt.xlabel("Value")
plt.ylabel("Frequency")




# 第一次量化
# quantized_data = quantize(data, b1)

quantized_data, dequantized_data = interval_based_quantize_dequantize(data, b1) # 使用基于区间的量化函数

# 第一次量化后的分布图
plt.subplot(1, 3, 2)
plt.hist(dequantized_data.numpy(), bins=100, color='green', alpha=0.7)
plt.title(f"Quantized to {b1} Levels (2^{n1})")
plt.xlabel("Value")
plt.ylabel("Frequency")

start_time = time.time()  # 记录第二次量化的开始时间

# # 第二次量化，还是进行均匀区间量化级别缩减
quantized_data_second, dequantized_data_second = interval_based_quantize_dequantize(dequantized_data, b2) # 使用基于区间的量化函数


#quantized_data_second, hist_data = dynamic_quantize_level_reduce(quantized_data, data, b2)
#quantized_data_second = dynamic_quantize_level_reduce_dequantize(quantized_data, data, b2)



# 计算动态量化的最终损失（均方误差）
final_error_dynamic = ((dequantized_data_second - data) ** 2).sum().item()
num_samples = data.size(0) # 这是样本数量
mean_squared_error_dynamic = final_error_dynamic / num_samples

# 最终量化后分布图
plt.subplot(1, 3, 3)
plt.hist(dequantized_data_second.numpy(), bins=100, color='red', alpha=0.7)
plt.title(f"Quantized to {b2} Levels (2^{n2})")
plt.xlabel("Value")
plt.ylabel("Frequency")
# X轴的范围是-b1 / 2, b1/ 2 -1
# plt.xlim(-half_range_b2, half_range_b2)

plt.tight_layout()
plt.show()



# 对比！
# ------------------------均匀量化--------------------------
# quantized_data_uniform = uniform_quantization(data, b2)
quantized_data_uniform, dequantized_data_uniform = interval_based_quantize_dequantize(data, b2) # 均匀量化
# quantized_data_uniform_mapped = map_to_target_range(quantized_data_uniform, 0, b1 - 1, -half_range_b2, half_range_b2)



# 计算均匀量化的最终损失（均方误差）
final_error_uniform_mapped = ((dequantized_data_uniform - data) ** 2).sum().item()
mean_squared_error_uniform_mapped = final_error_uniform_mapped / num_samples

# ------------------------均匀量化--------------------------




# 打印时间和最终均方误差
minimization_time = time.time() - start_time
print(f"Minimization to {b2} levels took {minimization_time:.4f} seconds.")
print(f"Final Mean Squared Error (本算法): {mean_squared_error_dynamic:.4f}")
print(f"Final Mean Squared Error (均匀): {mean_squared_error_uniform_mapped:.4f}")




# 保存量化后的数据
# 即保存quantized_data_second，这个是量化到[0, b2-1]的数据，需要映射到[-b2/2, b2/2-1]区间
# 如果用dequantized_data_second的话，因为反量化得到的是浮点数不满足后续所需的数据
quantized_data_second_mapped = map_to_target_range(
    quantized_data_second, 0, b2 - 1, -half_range(b2), half_range(b2) - 1
)

# 保存映射后的量化数据
output_file = "output/quantized_data_result.txt"  # 文件名

# 将量化数据转换为 numpy 格式
quantized_data_second_numpy = quantized_data_second_mapped.numpy()

# 使用 numpy.savetxt 保存
import numpy as np
np.savetxt(output_file, quantized_data_second_numpy, fmt="%.0f")

print(f"Quantized data saved to {output_file}")



# 将量化值和映射后的值组合成两列
quantized_data_second_numpy = quantized_data_second.numpy().astype(int)  # 整数量化值
quantized_data_second_mapped_numpy = quantized_data_second_mapped.numpy()  # 映射后的值

# 组合为二维数组，每行保存一个对应关系
correspondence_data = np.column_stack((quantized_data_second_numpy, quantized_data_second_mapped_numpy))

# 保存为文本文件
output_file = "output/quantized_data_correspondence.txt"
np.savetxt(output_file, correspondence_data, fmt="%d %.0f", header="Quantized_Value Mapped_Value", comments="")

print(f"Quantized correspondence data saved to {output_file}")






original_data_numpy = data.numpy()  # 原始数据
correspondence_data = np.column_stack((original_data_numpy, quantized_data_second_numpy, quantized_data_second_mapped_numpy)) # 将两个一维数组组合成二维数组，每行保存一个对应关系

# 保存带三列的数据
output_file = "output/quantized_data_with_original.txt"
np.savetxt(output_file, correspondence_data, fmt="%.6f %d %.0f", header="Original_Data Quantized_Value Mapped_Value", comments="")

print(f"Full correspondence data saved to {output_file}")
