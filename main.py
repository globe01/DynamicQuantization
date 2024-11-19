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





# 主程序
# data = generate_data()  # 生成数据

# data = generate_single_peak_data()  # 生成单峰均匀分布数据
# data = generate_double_peak_data()  # 生成双峰均匀分布数据
# data = generate_uniform_data()  # 生成均匀分布数据
# data = generate_exponential_data()  # 生成指数分布数据
# data = generate_multi_peak_data()  # 生成多峰分布数据
# data = generate_triangular_data()  # 生成三角分布数据
data = generate_poisson_data()  # 生成泊松分布数据
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

# 进行动态量化级别缩减
start_time = time.time()  # 记录开始时间
# quantized_data_second, hist_data = dynamic_quantize_level_reduce(quantized_data, data, b2)
quantized_data_second, hist_data = dynamic_quantize_level_reduce_dequantize(dequantized_data, data, b2)


# 全弄完之后，将当前的量化数据 Q 从[0, b1-1]映射到 [-b2 / 2, b2 / 2 - 1] 区间
# quantized_data_second = map_to_target_range(Q, 0, b1 - 1, -half_range(b2), half_range(b2) - 1)


# 计算动态量化的最终损失（均方误差）
final_error_dynamic = ((quantized_data_second - data) ** 2).sum().item()
num_samples = data.size(0)
mean_squared_error_dynamic = final_error_dynamic / num_samples

# 最终量化后分布图
plt.subplot(1, 3, 3)
plt.hist(quantized_data_second.numpy(), bins=100, color='red', alpha=0.7)
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
# quantized_data_uniform = interval_based_quantize(data, b2) # 均匀量化，自带反量化了
quantized_data_uniform, dequantized_data_uniform = interval_based_quantize_dequantize(data, b2) # 均匀量化，自带反量化了
# quantized_data_uniform_mapped = map_to_target_range(quantized_data_uniform, 0, b1 - 1, -half_range_b2, half_range_b2)

# 计算均匀量化的最终损失（均方误差）
final_error_uniform_mapped = ((dequantized_data_uniform - data) ** 2).sum().item()
mean_squared_error_uniform_mapped = final_error_uniform_mapped / num_samples

# ------------------------均匀量化--------------------------







# 动画绘制
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(-half_range_b2, half_range_b2)  # 设置x轴范围
ax.set_ylim(0, 5000)  # 设置y轴范围为0到5000

# 动画更新函数
def update(frame):
    ax.clear()  # 清除当前图形
    ax.hist(hist_data[frame], bins=100, color='r', alpha=0.7)
    ax.set_title(f"Quantization Step: {frame + 1}")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.set_xlim(-half_range_b2, half_range_b2)  # 设值x轴范围
    ax.set_ylim(0, 5000)  # 设置y轴范围，暂定5000

ani = FuncAnimation(fig, update, frames=len(hist_data), repeat=False) # 动画

# 保存为GIF
ani.save('quantization_process.gif', writer='imagemagick', fps=1)





# 打印时间和最终均方误差
minimization_time = time.time() - start_time
print(f"Minimization to {b2} levels took {minimization_time:.4f} seconds.")
print(f"Final Mean Squared Error (本算法): {mean_squared_error_dynamic:.4f}")
print(f"Final Mean Squared Error (均匀): {mean_squared_error_uniform_mapped:.4f}")


# Final Mean Squared Error (本算法): 18.4743
# Final Mean Squared Error (均匀): 0.2465
# 目前本算法的误差很大，是因为还没有反量化回到原来的范围，反量化待实现
