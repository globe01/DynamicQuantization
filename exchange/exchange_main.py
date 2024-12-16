import matplotlib.pyplot as plt
import time
import torch
import numpy as np
import os

from exchange_quantization_utils import (
    n1, n2, b1, b2, half_range, generate_data,
    interval_based_quantize, interval_based_quantize_dequantize, 
    interval_based_quantize_dequantize_2,
    interval_based_quantize_eachprint, interval_based_quantize_map,
    dynamic_quantize_level_reduce,
    dynamic_quantize_level_reduce_dequantize, map_to_target_range, 
    dynamic_quantize_level_reduce_dequantize_2, 
    new_dynamic_quantize
)

from exchange_randomDistribution import (
    generate_single_peak_data, generate_double_peak_data,
    generate_uniform_data, generate_exponential_data,
    generate_multi_peak_data, generate_triangular_data,
    generate_poisson_data, generate_extreme_values_data,
    generate_gaussian_with_uniform_noise
)



def main():
    # 1. 数据生成
    print("\n1. 生成测试数据...")

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
    print(f"生成数据大小: {data.shape}, 数据范围: [{data.min():.4f}, {data.max():.4f}]")

    # 2. 可视化设置
    plt.figure(figsize=(15, 5))
    
    # 原始数据分布图
    plt.subplot(1, 3, 1)
    plt.hist(data.numpy(), bins=100, color='blue', alpha=0.7)
    plt.title("Original Data Distribution")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    # 3. 执行量化过程
    print("\n2. 开始量化过程...")
    start_time = time.time()
    try:
        final_dequantized, hist_data, quantized_data, scale, bias = new_dynamic_quantize(
            data=data,
            initial_levels=b1,  # 16级
            target_levels=b2,   # 8级
            verbose=True        # 显示详细过程
        )
    except Exception as e:
        print(f"量化过程出错: {e}")
        return

    # 第一次量化结果可视化
    plt.subplot(1, 3, 2)
    plt.hist(hist_data[0], bins=100, color='green', alpha=0.7)
    plt.title(f"Quantized to {b1} Levels (2^{n1})")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    # 最终量化结果可视化
    plt.subplot(1, 3, 3)
    plt.hist(final_dequantized.numpy(), bins=100, color='red', alpha=0.7)
    plt.title(f"Quantized to {b2} Levels (2^{n2})")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    plt.tight_layout()
    
    # 4. 性能评估
    print("\n3. 计算性能指标...")
    # 动态量化误差计算
    final_error_dynamic = ((final_dequantized - data) ** 2).sum().item()
    num_samples = data.size(0)
    mean_squared_error_dynamic = final_error_dynamic / num_samples

    # 均匀量化对比实验
    uniform_quantized, uniform_dequantized = interval_based_quantize_dequantize(
        data=data,
        num_levels=b2  # 直接量化到8级
    )
    
    # 计算均匀量化误差
    final_error_uniform = ((uniform_dequantized - data) ** 2).sum().item()
    mean_squared_error_uniform = final_error_uniform / num_samples

    # 5. 结果展示
    minimization_time = time.time() - start_time
    print(f"\n量化性能评估:")
    print(f"量化用时: {minimization_time:.4f} 秒")
    print(f"动态量化均方误差: {mean_squared_error_dynamic:.6f}")
    print(f"均匀量化均方误差: {mean_squared_error_uniform:.6f}")
    print(f"性能提升: {((mean_squared_error_uniform - mean_squared_error_dynamic) / mean_squared_error_uniform * 100):.2f}%")


    # 6. 数据保存
    print("\n4. 保存结果...")
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 保存量化数据
    output_file = os.path.join(output_dir, "dynamic_quantized_data.txt")
    try:
        # 保存未经反量化的数据，此时还是整数
        np.savetxt(output_file, quantized_data.numpy(), fmt="%d",
                header="Dynamic Quantized Data", comments="")
        print(f"量化数据已保存至: {output_file}")

        # 生成并保存code book
        # 获取唯一的量化值并排序
        unique_quantized = torch.unique(quantized_data).numpy().astype(int)
        unique_quantized.sort()
        
        # 创建0-7的目标量化值
        target_levels = np.arange(b2)  # b2=8时，生成[0,1,2,3,4,5,6,7]
        
        # 计算映射关系
        if len(unique_quantized) != len(target_levels):
            raise ValueError(f"量化值数量 {len(unique_quantized)} 与目标级别数 {len(target_levels)} 不匹配")
        
        # 创建code book数组
        code_book = np.column_stack((
            target_levels,                      # 目标量化值（0-7）
            unique_quantized,                   # 当前量化值
            unique_quantized - target_levels,   # 映射关系（差值）
            np.full_like(target_levels, scale, dtype=float),  # scale值
            np.full_like(target_levels, bias, dtype=float)    # bias值
        ))
        
        # 保存code book
        code_book_file = os.path.join(output_dir, "code_book.txt")
        np.savetxt(code_book_file, code_book,
                fmt=['%d', '%d', '%d', '%.6f', '%.6f'],  # 前三列用整数格式
                header="Target_Level Quantized_Value Mapping_Difference Scale Bias",
                comments="")
        print(f"Code book已保存至: {code_book_file}")
        
        # 打印code book内容
        print("\nCode Book内容:")
        print("目标值  量化值   映射关系   Scale     Bias")
        for i in range(len(target_levels)):
            print(f"{int(code_book[i,0]):3d}    {int(code_book[i,1]):3d}     {int(code_book[i,2]):3d}       {code_book[i,3]:8.4f}  {code_book[i,4]:8.4f}")
            
    except Exception as e:
        print(f"保存数据时出错: {e}")

    plt.show()

if __name__ == "__main__":
    main()