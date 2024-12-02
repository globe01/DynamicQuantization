import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import torch

from quantization_utils import (
    n1, n2, b1, b2, half_range_b2, half_range, generate_data,
    interval_based_quantize, interval_based_quantize_dequantize,
    interval_based_quantize_eachprint, interval_based_quantize_map,
    dynamic_quantize_level_reduce,
    dynamic_quantize_level_reduce_dequantize, map_to_target_range, dynamic_quantize_level_reduce_dequantize_2
)

# from distribution import (
#     generate_single_peak_data, generate_double_peak_data,
#     generate_uniform_data, generate_exponential_data,
#     generate_multi_peak_data, generate_triangular_data,
#     generate_poisson_data, generate_extreme_values_data,
#     generate_gaussian_with_uniform_noise
# )

# 纯随机种子
from randomDistribution import (
    generate_single_peak_data, generate_double_peak_data,   # 生成单峰正态分布数据
    generate_uniform_data, generate_exponential_data,       # 生成均匀分布数据、指数分布数据
    generate_multi_peak_data, generate_triangular_data,     # 生成多峰分布数据、三角分布数据
    generate_poisson_data, generate_extreme_values_data,     # 生成泊松分布数据、极端值分布数据
    generate_gaussian_with_uniform_noise                    # 生成高斯分布+均匀噪声数据
)


# ------------------------真实llama2参数测试--------------------------------------------------
# from extract_weights import extract_weights_in_chunks
#
#
# # 调用函数生成 `all_weights_tensor`
# checkpoint_path = "C:/Users/globe/.llama/checkpoints/Llama-2-7b/consolidated.00.pth"
# target_keys = ["attention.wq.weight", "attention.wk.weight", "attention.wv.weight", "attention.wo.weight",
#                "feed_forward.w1.weight", "feed_forward.w2.weight", "feed_forward.w3.weight"]
#
# data = extract_weights_in_chunks(checkpoint_path, target_keys)
#
# # 后续对 `all_weights_tensor` 的量化操作
# print("合并后的权重数据形状:", data.shape if data is not None else "无匹配权重")
#
# if data is None:
#     raise ValueError("未找到任何匹配的权重，请检查 target_keys 和模型文件结构。")
#
# # 将 data 转换为 float32 以便用于绘图
# data = data.to(torch.float32)
#
# # 假设 `data` 为大权重张量
# chunk_size = 1000000  # 每个分块的大小，根据内存大小调整
# num_chunks = (data.numel() + chunk_size - 1) // chunk_size  # 计算分块数量
#
# quantized_chunks = []
# ------------------------------------------------------------------------------------------



# --------------------------------各种不同分布的测试数据-----------------------------------------
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
# ------------------------------------------------------------------------------------------


# ---------------------------------------原始数据分布图----------------------------------------
# 画图
plt.figure(figsize=(15, 5))  # 画布总的大小
# 先画原始数据分布图
plt.subplot(1, 3, 1)
plt.hist(data.numpy(), bins=100, color='blue', alpha=0.7)
plt.title("Original Data Distribution")
plt.xlabel("Value")
plt.ylabel("Frequency")
# ------------------------------------------------------------------------------------------



# --------------------------------------第1次量化（均匀量化）-----------------------------------
# quantized_data = quantize(data, b1)

quantized_data, dequantized_data = interval_based_quantize_dequantize(data, b1) # 使用基于区间的量化函数

# 第一次量化后的分布图
plt.subplot(1, 3, 2)
plt.hist(dequantized_data.numpy(), bins=100, color='green', alpha=0.7)
plt.title(f"Quantized to {b1} Levels (2^{n1})")
plt.xlabel("Value")
plt.ylabel("Frequency")
# -------------------------------------------------------------------------------------------



# --------------------------------------第2次量化（动态量化）------------------------------------
# 进行动态量化级别缩减
start_time = time.time()  # 记录开始时间

# quantized_data_second, hist_data = dynamic_quantize_level_reduce(quantized_data, data, b2)
# quantized_data_second, hist_data = dynamic_quantize_level_reduce_dequantize(dequantized_data, data, b2)
quantized_data_second, hist_data = dynamic_quantize_level_reduce_dequantize_2(dequantized_data, data, b2)

# 全弄完之后，将当前的量化数据 Q 从[0, b1-1]映射到 [-b2 / 2, b2 / 2 - 1] 区间
# quantized_data_second = map_to_target_range(Q, 0, b1 - 1, -half_range(b2), half_range(b2) - 1)




# 第2次量化后，即最终分布图
plt.subplot(1, 3, 3)
plt.hist(quantized_data_second.numpy(), bins=100, color='red', alpha=0.7)
plt.title(f"Quantized to {b2} Levels (2^{n2})")
plt.xlabel("Value")
plt.ylabel("Frequency")
# X轴的范围是-b1 / 2, b1/ 2 -1
# plt.xlim(-half_range_b2, half_range_b2)

plt.tight_layout()
plt.show()
# -------------------------------------------------------------------------------------------


# --------------------------------------计算最终均方误差----------------------------------------
# 计算动态量化的最终损失（均方误差）
final_error_dynamic = ((quantized_data_second - data) ** 2).sum().item()
num_samples = data.size(0)
mean_squared_error_dynamic = final_error_dynamic / num_samples
# -------------------------------------------------------------------------------------------







# -----------------------------------------对比实验-------------------------------------------
# ------------------------再来一个全部都是均匀量化的作为对比---------------------------------------
# quantized_data_uniform = uniform_quantization(data, b2)
# quantized_data_uniform = interval_based_quantize(data, b2) # 均匀量化，自带反量化了
quantized_data_uniform, dequantized_data_uniform = interval_based_quantize_dequantize(data, b2) # 均匀量化，自带反量化了
# quantized_data_uniform_mapped = map_to_target_range(quantized_data_uniform, 0, b1 - 1, -half_range_b2, half_range_b2)

# 计算均匀量化的最终损失（均方误差）
final_error_uniform_mapped = ((dequantized_data_uniform - data) ** 2).sum().item()
mean_squared_error_uniform_mapped = final_error_uniform_mapped / num_samples

# -------------------------------------------------------------------------------------------






# ---------------------------------------绘制gif动画--------------------------------------------
# fig, ax = plt.subplots(figsize=(8, 6))
# ax.set_xlim(-half_range_b2, half_range_b2)  # 设置x轴范围
# ax.set_ylim(0, 5000)  # 设置y轴范围为0到5000
#
# # 动画更新函数
# def update(frame):
#     ax.clear()  # 清除当前图形
#     ax.hist(hist_data[frame], bins=100, color='r', alpha=0.7)
#     ax.set_title(f"Quantization Step: {frame + 1}")
#     ax.set_xlabel("Value")
#     ax.set_ylabel("Frequency")
#     ax.set_xlim(-half_range_b2, half_range_b2)  # 设值x轴范围
#     ax.set_ylim(0, 5000)  # 设置y轴范围，暂定5000
#
# ani = FuncAnimation(fig, update, frames=len(hist_data), repeat=False) # 动画
#
# # 保存为GIF
# ani.save('quantization_process.gif', writer='imagemagick', fps=1)
# -------------------------------------------------------------------------------------------




# ------------------------------------打印时间和最终均方误差-------------------------------------
minimization_time = time.time() - start_time
print(f"Minimization to {b2} levels took {minimization_time:.4f} seconds.")
print(f"Final Mean Squared Error (本算法): {mean_squared_error_dynamic:.4f}")
print(f"Final Mean Squared Error (均匀): {mean_squared_error_uniform_mapped:.4f}")

# Final Mean Squared Error (本算法): 0.2302
# Final Mean Squared Error (均匀): 0.2465
# -------------------------------------------------------------------------------------------













# -----------------------------------------保存量化后的数据-------------------------------------

# 即保存quantized_data_second，这个是量化到[0, b2-1]的数据，需要映射到[-b2/2, b2/2-1]区间
# 如果用dequantized_data_second的话，因为反量化得到的是浮点数不满足后续所需的数据
quantized_data_second_mapped = map_to_target_range(
    quantized_data_second, 0, b2 - 1, -half_range(b2), half_range(b2) - 1
)

# 保存映射后的量化数据
output_file = "output/dynamic_quantized_data_result.txt"  # 文件名

# 将量化数据转换为 numpy 格式
quantized_data_second_numpy = quantized_data_second_mapped.numpy()

# 使用 numpy.savetxt 保存
import numpy as np
np.savetxt(output_file, quantized_data_second_numpy, fmt="%.0f")

print(f"Quantized data saved to {output_file}")



# 将量化值和映射后的值组合成两列
quantized_data_second_numpy = quantized_data_second.numpy().astype(int)  # 整数量化值
quantized_data_second_mapped_numpy = quantized_data_second_mapped.numpy()  # 映射后的值

# 获取所有唯一的量化值
unique_quantized_values = np.unique(quantized_data_second_numpy)

# 计算 Scale 和 Bias
scale = (quantized_data_second_mapped_numpy.max() - quantized_data_second_mapped_numpy.min()) / (b2 - 1)
bias = quantized_data_second_mapped_numpy.min()

# 计算每个量化值对应的映射值（使用 Scale 和 Bias）
mapped_values = [q * scale + bias for q in unique_quantized_values]

# 创建完整的 Code Book，包括量化值、映射值、Scale 和 Bias
code_book = np.column_stack((
    unique_quantized_values,         # Quantized_Value
    mapped_values,                   # Mapped_Value
    np.full_like(unique_quantized_values, scale),  # Scale (重复值)
    np.full_like(unique_quantized_values, bias)    # Bias (重复值)
))

# 保存 Code Book 到文件
output_file_codebook = "output/dynamic_code_book_with_scale_bias.txt"
np.savetxt(
    output_file_codebook, code_book,
    fmt="%d %.f %.f %.f",
    header="Quantized_Value Mapped_Value Scale Bias",
    comments=""
)

print(f"Code Book with scale and bias saved to {output_file_codebook}")



# 保存带三列的数据
# original_data_numpy = data.numpy()  # 原始数据
# correspondence_data = np.column_stack((original_data_numpy, quantized_data_second_numpy, quantized_data_second_mapped_numpy)) # 将两个一维数组组合成二维数组，每行保存一个对应关系
# output_file = "output/dynamic_quantized_data_with_original.txt"
# np.savetxt(output_file, correspondence_data, fmt="%.6f %d %.0f", header="Original_Data Quantized_Value Mapped_Value", comments="")
#
# print(f"Full correspondence data saved to {output_file}")

# -------------------------------------------------------------------------------------------