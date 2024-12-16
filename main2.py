# llama2真实的每层权重数据

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import os

from quantization_utils import (
    interval_based_quantize_dequantize,
    dynamic_quantize_level_reduce_dequantize_2,
    b1, b2
)


def analyze_weight_distribution(weight_tensor, name):
    """分析权重分布的统计信息"""
    stats = {
        'mean': float(torch.mean(weight_tensor)),
        'std': float(torch.std(weight_tensor)),
        'min': float(torch.min(weight_tensor)),
        'max': float(torch.max(weight_tensor)),
        'sparsity': float((weight_tensor == 0).sum() / weight_tensor.numel()),
        'shape': weight_tensor.shape,
        'num_elements': weight_tensor.numel()
    }
    return stats


def plot_weight_distribution(weight_tensor, name, save_path=None):
    """绘制并保存权重分布图"""
    weight_tensor = weight_tensor.to(torch.float32)
    plt.figure(figsize=(10, 6))
    plt.hist(weight_tensor.cpu().numpy().flatten(), bins=100, density=True)
    plt.title(f'Weight Distribution: {name}')
    plt.xlabel('Weight Value')
    plt.ylabel('Density')
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f'{save_path}/{name}_distribution.png')
        plt.close()
    else:
        plt.show()


def process_single_layer(weight, key, chunk_size=1000000):
    """处理单层权重"""
    print(f"\nProcessing layer: {key}")

    # 如果是 BFloat16，转换为 Float32
    if weight.dtype == torch.bfloat16:
        print(f"Converting {key} from BFloat16 to Float32 for compatibility.")
        weight = weight.to(torch.float32)

    # 收集并打印统计信息
    stats = analyze_weight_distribution(weight, key)
    print(f"Shape: {stats['shape']}")
    print(f"Mean: {stats['mean']:.6f}")
    print(f"Std: {stats['std']:.6f}")
    print(f"Min: {stats['min']:.6f}")
    print(f"Max: {stats['max']:.6f}")
    print(f"Sparsity: {stats['sparsity']:.2%}")

    # 绘制原始分布图
    plot_weight_distribution(weight, f"{key}_original", "weight_distributions")

    # 展平权重
    flattened_weight = weight.view(-1)

    # 添加：直接使用均匀量化到8个level的结果
    print("Performing uniform quantization to 8 levels...")
    quantized_data_uniform, dequantized_data_uniform = interval_based_quantize_dequantize(flattened_weight, b2)
    uniform_error = ((dequantized_data_uniform - flattened_weight) ** 2).mean().item()
    print(f"Uniform quantization mean squared error: {uniform_error:.4f}")

    # 第一次量化（均匀量化）
    print("Performing first quantization step...")
    start_time = time.time()

    # 分块处理权重
    chunks_q = []
    chunks_dq = []
    num_chunks = (flattened_weight.numel() + chunk_size - 1) // chunk_size

    for i in tqdm(range(num_chunks), desc="First quantization"):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, flattened_weight.numel())
        chunk = flattened_weight[start_idx:end_idx]
        quantized_chunk, dequantized_chunk = interval_based_quantize_dequantize(chunk, b1)
        chunks_q.append(quantized_chunk)
        chunks_dq.append(dequantized_chunk)

    quantized_data = torch.cat(chunks_q)
    dequantized_data = torch.cat(chunks_dq)
    first_quant_time = time.time() - start_time
    print(f"First quantization completed in {first_quant_time:.2f} seconds")

    # 第二次量化（动态量化）
    print("Performing second quantization step...")
    start_time = time.time()
    final_quantized_data, hist_data, dynamic_quantized_data = dynamic_quantize_level_reduce_dequantize_2(
        dequantized_data, quantized_data, b2
    )
    second_quant_time = time.time() - start_time
    print(f"Second quantization completed in {second_quant_time:.2f} seconds")

    # 计算动态量化的误差
    dynamic_error = ((final_quantized_data - flattened_weight) ** 2).mean().item()
    print(f"Dynamic quantization mean squared error: {dynamic_error:.4f}")

    # 保存误差结果到txt文件
    os.makedirs("quantization_results", exist_ok=True)
    with open(f"quantization_results/{key}_errors.txt", "w") as f:
        f.write("Dynamic MSE  Uniform MSE\n")
        f.write(f"{dynamic_error:.4f} {uniform_error:.4f}")

    # 绘制量化结果对比图
    plot_quantization_results(
        flattened_weight, # 显示原始数据
        dequantized_data,
        final_quantized_data,
        key,
        dequantized_data_uniform  # 新增这个参数
    )

    return {
        'stats': stats,
        'dynamic_quantized_data': dynamic_quantized_data,
        'final_quantized_data': final_quantized_data,
        'quantization_stats': {
            'first_quant_time': first_quant_time,
            'second_quant_time': second_quant_time,
            'dynamic_error': dynamic_error,
            'uniform_error': uniform_error
        }
    }


def plot_quantization_results(original, first_quant, second_quant, layer_name, uniform_quant):
    """绘制量化结果对比图"""
    plt.figure(figsize=(20, 5))  # 加宽图片以容纳4个子图

    plt.subplot(1, 4, 1)
    plt.hist(original.numpy(), bins=100, color='blue', alpha=0.7)
    plt.title("Original Weight Distribution")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    plt.subplot(1, 4, 2)
    plt.hist(first_quant.numpy(), bins=100, color='green', alpha=0.7)
    plt.title(f"First Quantization ({b1} levels)")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    plt.subplot(1, 4, 3)
    plt.hist(second_quant.numpy(), bins=100, color='red', alpha=0.7)
    plt.title(f"Dynamic Quantization ({b2} levels)")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    # 添加均匀量化的结果
    plt.subplot(1, 4, 4)
    plt.hist(uniform_quant.numpy(), bins=100, color='purple', alpha=0.7)
    plt.title(f"Uniform Quantization ({b2} levels)")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    plt.tight_layout()
    os.makedirs("quantization_results", exist_ok=True)
    plt.savefig(f"quantization_results/{layer_name}_quantization.png")
    plt.close()


def process_model_weights(checkpoint_path):
    """逐层处理模型权重"""
    target_keys = [
        "attention.wq.weight", "attention.wk.weight", "attention.wv.weight", "attention.wo.weight",
        "feed_forward.w1.weight", "feed_forward.w2.weight", "feed_forward.w3.weight"
    ]

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # 防止目录不存在
    # weights_dir = "model-weights"
    # os.makedirs(weights_dir, exist_ok=True)

    results = {}
    for key, weight in checkpoint.items():
        if any(tk in key for tk in target_keys):
            try:
                print(f"\nProcessing weights for: {key}")

                result = process_single_layer(weight, key)
                results[key] = result

                # 保存模型权重pth文件
                # save_path = os.path.join(weights_dir, f'quantized_weights_{key.replace(".", "_")}.pth')
                # torch.save(result, save_path)
                # print(f"Saved results to {save_path}")

                # 释放内存
                del result
                torch.cuda.empty_cache()  # 如果使用了GPU

            except Exception as e:
                print(f"Error processing layer {key}: {str(e)}")
                continue

    return results


def main():
    """主程序"""
    torch.manual_seed(42)
    np.random.seed(42)

    checkpoint_path = "C:/Users/globe/.llama/checkpoints/Llama-2-7b/consolidated.00.pth"

    try:
        results = process_model_weights(checkpoint_path)
        print("\nProcessing completed successfully!")

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise


if __name__ == "__main__":
    main()