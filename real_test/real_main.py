import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from typing import List, Dict, Optional, Tuple
import time

from real_quantization_utils import (
    n1, n2, b1, b2, half_range, generate_data,
    interval_based_quantize, interval_based_quantize_dequantize,
    interval_based_quantize_dequantize_2,
    interval_based_quantize_eachprint, interval_based_quantize_map,
    dynamic_quantize_level_reduce,
    dynamic_quantize_level_reduce_dequantize, map_to_target_range,
    dynamic_quantize_level_reduce_dequantize_2,
    new_dynamic_quantize
)


def extract_layer_weights(checkpoint_path: str, layer_num: int = 0) -> Dict[str, torch.Tensor]:
    """
    从LLaMA-2检查点文件中提取指定层的权重。

    Args:
        checkpoint_path: 检查点文件路径
        layer_num: 要提取的层号

    Returns:
        包含该层权重的字典
    """
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 定义要提取的键——只包含涉及矩阵乘法的权重
    target_keys = [
        f'layers.{layer_num}.attention.wq.weight',  # 查询矩阵乘法
        f'layers.{layer_num}.attention.wk.weight',  # 键矩阵乘法
        f'layers.{layer_num}.attention.wv.weight',  # 值矩阵乘法
        f'layers.{layer_num}.attention.wo.weight',  # 输出矩阵乘法
        f'layers.{layer_num}.feed_forward.w1.weight',  # FFN第一个矩阵乘法
        f'layers.{layer_num}.feed_forward.w2.weight',  # FFN第二个矩阵乘法
        f'layers.{layer_num}.feed_forward.w3.weight'  # FFN第三个矩阵乘法
    ]

    # 提取权重
    weights = {}
    for key in target_keys:
        if key in checkpoint:
            weights[key] = checkpoint[key].float()  # 转换为float32以便后续处理
        else:
            print(f"Warning: Key {key} not found in checkpoint")

    return weights


def quantize_and_evaluate(weight_tensor: torch.Tensor,
                          initial_levels: int,
                          target_levels: int,
                          name: str,
                          save_dir: str) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    """
    对权重进行量化并评估结果。

    Args:
        weight_tensor: 输入权重张量
        initial_levels: 初始量化级别（第一步量化）
        target_levels: 目标量化级别（第二步量化）
        name: 权重名称
        save_dir: 结果保存目录

    Returns:
        dynamic_result: 动态量化结果
        uniform_result: 均匀量化结果
        dynamic_error: 动态量化误差
        uniform_error: 均匀量化误差
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 动态量化
    final_dequantized, hist_data, quantized_data, scale, bias = new_dynamic_quantize(
        data=weight_tensor,
        initial_levels=initial_levels,
        target_levels=target_levels,
        verbose=True
    )

    # 均匀量化对比（直接一步到位成8个level的均匀量化）
    uniform_quantized, uniform_dequantized = interval_based_quantize_dequantize(
        data=weight_tensor,
        num_levels=target_levels
    )

    # 计算误差
    dynamic_error = ((final_dequantized - weight_tensor) ** 2).mean().item()
    uniform_error = ((uniform_dequantized - weight_tensor) ** 2).mean().item()

    # 保存量化结果可视化
    plot_quantization_results(
        original=weight_tensor,
        first_quant=torch.tensor(hist_data[0]),
        second_quant=final_dequantized,
        layer_name=name,
        uniform_quant=uniform_dequantized,
        save_dir=save_dir
    )

    # 保存量化数据和codebook
    save_quantization_results(
        name=name,
        quantized_data=quantized_data,
        dynamic_error=dynamic_error,
        uniform_error=uniform_error,
        scale=scale,
        bias=bias,
        target_levels=target_levels,
        save_dir=save_dir
    )

    return final_dequantized, uniform_dequantized, dynamic_error, uniform_error


def plot_quantization_results(original: torch.Tensor,
                              first_quant: torch.Tensor,
                              second_quant: torch.Tensor,
                              layer_name: str,
                              uniform_quant: torch.Tensor,
                              save_dir: str):
    """绘制量化结果对比图"""
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 4, 1)
    plt.hist(original.numpy().flatten(), bins=100, color='blue', alpha=0.7)
    plt.title("Original Weight Distribution")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    plt.subplot(1, 4, 2)
    plt.hist(first_quant.flatten(), bins=100, color='green', alpha=0.7)
    plt.title(f"First Quantization (16 levels)")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    plt.subplot(1, 4, 3)
    plt.hist(second_quant.numpy().flatten(), bins=100, color='red', alpha=0.7)
    plt.title(f"Dynamic Quantization (8 levels)")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    plt.subplot(1, 4, 4)
    plt.hist(uniform_quant.numpy().flatten(), bins=100, color='purple', alpha=0.7)
    plt.title(f"Uniform Quantization (8 levels)")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{layer_name}_quantization.png"))
    plt.close()


# def save_quantization_results(name: str,
#                               quantized_data: torch.Tensor,
#                               dynamic_error: float,
#                               uniform_error: float,
#                               scale: float,
#                               bias: float,
#                               target_levels: int,
#                               save_dir: str):
#     """保存量化结果，包括误差、量化数据和codebook"""
#     # 保存误差结果
#     with open(os.path.join(save_dir, f"{name}_errors.txt"), "w") as f:
#         f.write("Dynamic MSE  Uniform MSE\n")
#         f.write(f"{dynamic_error:.6f} {uniform_error:.6f}")
#
#     try:
#         # 保存量化数据
#         output_file = os.path.join(save_dir, f"{name}_quantized_data.txt")
#         np.savetxt(output_file, quantized_data.numpy(), fmt="%d",
#                    header="Dynamic Quantized Data", comments="")
#         print(f"量化数据已保存至: {output_file}")
#
#         # 获取唯一的量化值并排序
#         unique_quantized = torch.unique(quantized_data).numpy().astype(int)
#         unique_quantized.sort()
#
#         # 创建目标量化值
#         target_levels_array = np.arange(target_levels)  # 生成[0,1,2,3,4,5,6,7]
#
#         # 检查量化值数量
#         if len(unique_quantized) != len(target_levels_array):
#             raise ValueError(f"量化值数量 {len(unique_quantized)} 与目标级别数 {len(target_levels_array)} 不匹配")
#
#         # 创建code book数组
#         code_book = np.column_stack((
#             target_levels_array,  # 目标量化值
#             unique_quantized,  # 当前量化值
#             unique_quantized - target_levels_array,  # 映射关系（差值）
#             np.full_like(target_levels_array, scale, dtype=float),  # scale值
#             np.full_like(target_levels_array, bias, dtype=float)  # bias值
#         ))
#
#         # 保存code book
#         code_book_file = os.path.join(save_dir, f"{name}_code_book.txt")
#         np.savetxt(code_book_file, code_book,
#                    fmt=['%d', '%d', '%d', '%.6f', '%.6f'],  # 前三列用整数格式
#                    header="Target_Level Quantized_Value Mapping_Difference Scale Bias",
#                    comments="")
#         print(f"Code book已保存至: {code_book_file}")
#
#         # 打印code book内容
#         print(f"\nCode Book内容 ({name}):")
#         print("目标值  量化值   映射关系   Scale     Bias")
#         for i in range(len(target_levels_array)):
#             print(
#                 f"{int(code_book[i, 0]):3d}    {int(code_book[i, 1]):3d}     {int(code_book[i, 2]):3d}       {code_book[i, 3]:8.4f}  {code_book[i, 4]:8.4f}")
#
#         # 新增，便于后续使用
#         # 创建映射字典：从原始量化值到目标值(0-7)的映射
#         mapping_dict = {orig: target for target, orig in zip(target_levels_array, unique_quantized)}
#
#         # 创建转换后的量化数据
#         quantized_data_mapped = torch.tensor([mapping_dict[val.item()] for val in quantized_data])
#
#         # 平移到[-4, 3]范围
#         quantized_data_shifted = quantized_data_mapped - 4
#
#         # 保存转换后的量化数据
#         transformed_output_file = os.path.join(save_dir, f"{name}_quantized_data_after_transformation.txt")
#         np.savetxt(transformed_output_file, quantized_data_shifted.numpy(), fmt="%d",
#                    header="Transformed Quantized Data (Range: -4 to 3)", comments="")
#         print(f"转换后的量化数据已保存至: {transformed_output_file}")
#
#
#     except Exception as e:
#         print(f"保存数据时出错: {e}")


def save_quantization_results(name: str,
                              quantized_data: torch.Tensor,
                              dynamic_error: float,
                              uniform_error: float,
                              scale: float,
                              bias: float,
                              target_levels: int,
                              save_dir: str):
    """保存量化结果，包括误差、量化数据和codebook"""
    try:
        # 保存误差结果
        with open(os.path.join(save_dir, f"{name}_errors.txt"), "w") as f:
            f.write("Dynamic MSE  Uniform MSE\n")
            f.write(f"{dynamic_error:.6f} {uniform_error:.6f}")
        print(f"误差数据已保存")

        # 保存原始量化数据
        try:
            output_file = os.path.join(save_dir, f"{name}_quantized_data.txt")
            np.savetxt(output_file, quantized_data.numpy(), fmt="%d",
                       header="Dynamic Quantized Data", comments="")
            print(f"原始量化数据已保存至: {output_file}")
        except Exception as e:
            print(f"保存原始量化数据时出错: {e}")
            return

        # 获取唯一的量化值并排序
        try:
            unique_quantized = torch.unique(quantized_data).numpy().astype(int)
            unique_quantized.sort()
            target_levels_array = np.arange(target_levels)

            if len(unique_quantized) != len(target_levels_array):
                print(f"警告: 量化值数量 {len(unique_quantized)} 与目标级别数 {len(target_levels_array)} 不匹配")
                print(f"唯一量化值: {unique_quantized}")
                print(f"目标级别: {target_levels_array}")
                return
        except Exception as e:
            print(f"处理唯一量化值时出错: {e}")
            return

        # 创建和保存 code book
        try:
            code_book = np.column_stack((
                target_levels_array,
                unique_quantized,
                unique_quantized - target_levels_array,
                np.full_like(target_levels_array, scale, dtype=float),
                np.full_like(target_levels_array, bias, dtype=float)
            ))

            code_book_file = os.path.join(save_dir, f"{name}_code_book.txt")
            np.savetxt(code_book_file, code_book,
                       fmt=['%d', '%d', '%d', '%.6f', '%.6f'],
                       header="Target_Level Quantized_Value Mapping_Difference Scale Bias",
                       comments="")
            print(f"Code book已保存至: {code_book_file}")
        except Exception as e:
            print(f"创建或保存code book时出错: {e}")
            return

        # 创建映射字典和转换后的量化数据
        try:
            # 创建映射字典
            mapping_dict = {int(orig): int(target) for target, orig in zip(target_levels_array, unique_quantized)}

            # 将量化数据转换为numpy数组进行处理
            quantized_numpy = quantized_data.numpy()

            # 使用numpy的vectorize功能创建映射函数
            vectorized_map = np.vectorize(lambda x: mapping_dict[int(x)])

            # 应用映射
            quantized_mapped = vectorized_map(quantized_numpy)

            # 进行平移操作
            quantized_shifted = quantized_mapped - (target_levels // 2)

            # 保存转换后的量化数据
            transformed_output_file = os.path.join(save_dir, f"{name}_quantized_data_after_transformation.txt")
            np.savetxt(transformed_output_file, quantized_shifted, fmt="%d",
                       header=f"Transformed Quantized Data (Range: -{target_levels // 2} to {target_levels // 2 - 1})",
                       comments="")
            print(f"转换后的量化数据已保存至: {transformed_output_file}")

            # 打印一些统计信息用于调试
            print(f"\n数据统计:")
            print(f"转换前的唯一值: {np.unique(quantized_numpy).tolist()}")
            print(f"映射后的唯一值: {np.unique(quantized_mapped).tolist()}")
            print(f"平移后的唯一值: {np.unique(quantized_shifted).tolist()}")
        except Exception as e:
            print(f"创建或保存转换后的量化数据时出错: {e}")
            print(f"错误详情: {str(e)}")
            return

    except Exception as e:
        print(f"保存量化结果时发生错误: {e}")



# # 单独处理一层的main
# def main():
#     start_time = time.time()
#     # 配置参数
#     checkpoint_path = "C:/Users/globe/.llama/checkpoints/Llama-2-7b/consolidated.00.pth"
#     save_dir = "quantization_results"
#     layer_num = 0  # 层数从0开始，0-31一共32层
#
#     # 提取权重
#     weights = extract_layer_weights(checkpoint_path, layer_num)
#     if not weights:
#         raise ValueError("未能成功提取权重")
#
#     # 对每个权重矩阵进行量化
#     results = {}
#     for name, weight in weights.items():
#         print(f"\n处理权重: {name}")
#         print(f"权重形状: {weight.shape}")
#
#         # 量化并评估
#         dynamic_result, uniform_result, dynamic_error, uniform_error = quantize_and_evaluate(
#             weight_tensor=weight,
#             initial_levels=16,  # 2^4
#             target_levels=8,  # 2^3
#             name=name.replace(".", "_"),
#             save_dir=save_dir
#         )
#
#         # 存储结果
#         results[name] = {
#             'dynamic_result': dynamic_result,
#             'uniform_result': uniform_result,
#             'dynamic_error': dynamic_error,
#             'uniform_error': uniform_error
#         }
#
#         # 打印性能对比
#         improvement = ((uniform_error - dynamic_error) / uniform_error * 100)
#         print(f"\n量化性能评估 ({name}):")
#         print(f"动态量化均方误差: {dynamic_error:.6f}")
#         print(f"均匀量化均方误差: {uniform_error:.6f}")
#         print(f"性能提升: {improvement:.2f}%")
#
#     print("\n量化完成！结果已保存到", save_dir)
#     end_time = time.time()
#
#     print(f"Second quantization completed in {end_time - start_time:.2f} seconds")


# 批量处理所有层的main
def main():
    total_start_time = time.time()
    # 配置参数
    checkpoint_path = "C:/Users/globe/.llama/checkpoints/Llama-2-7b/consolidated.00.pth"

    # 为每一层创建单独的结果目录
    for layer_num in range(1):  # 0-31，共32层
        layer_start_time = time.time()
        print(f"\n{'=' * 50}")
        print(f"开始处理第 {layer_num} 层 (总共32层)")
        print(f"{'=' * 50}")

        # 为每一层创建单独的保存目录
        save_dir = f"quantization_results/layer_{layer_num:02d}"
        os.makedirs(save_dir, exist_ok=True)

        # 提取权重
        weights = extract_layer_weights(checkpoint_path, layer_num)
        if not weights:
            print(f"Warning: 第 {layer_num} 层未能成功提取权重，跳过该层")
            continue

        # 对每个权重矩阵进行量化
        results = {}
        for name, weight in weights.items():
            print(f"\n处理权重: {name}")
            print(f"权重形状: {weight.shape}")

            # 量化并评估
            dynamic_result, uniform_result, dynamic_error, uniform_error = quantize_and_evaluate(
                weight_tensor=weight,
                initial_levels=16,  # 2^4
                target_levels=8,  # 2^3
                name=name.replace(".", "_"),
                save_dir=save_dir
            )

            # 存储结果
            results[name] = {
                'dynamic_result': dynamic_result,
                'uniform_result': uniform_result,
                'dynamic_error': dynamic_error,
                'uniform_error': uniform_error
            }

            # 打印性能对比
            improvement = ((uniform_error - dynamic_error) / uniform_error * 100)
            print(f"\n量化性能评估 ({name}):")
            print(f"动态量化均方误差: {dynamic_error:.6f}")
            print(f"均匀量化均方误差: {uniform_error:.6f}")
            print(f"性能提升: {improvement:.2f}%")

        # 每层完成后打印进度
        layer_end_time = time.time()
        layer_time = layer_end_time - layer_start_time
        print(f"\n第 {layer_num} 层处理完成！")
        print(f"本层处理时间: {layer_time:.2f} 秒")
        print(f"当前进度: {layer_num + 1}/32 层 ({((layer_num + 1) / 32 * 100):.1f}%)")

    # 所有层处理完成后的总结
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    print("\n" + "=" * 50)
    print("所有层处理完成！")
    print(f"总处理时间: {total_time:.2f} 秒")
    print(f"平均每层处理时间: {total_time / 32:.2f} 秒")
    print("=" * 50)


if __name__ == "__main__":
    main()