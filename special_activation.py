import torch
import matplotlib.pyplot as plt
import os
import numpy as np
import re
from typing import List, Dict, Optional, Tuple
import time
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
from torch.utils.hooks import RemovableHandle
from tqdm import tqdm
import scipy.stats as stats
from scipy.stats import skew, kurtosis
import warnings

from real_quantization_utils import (
    interval_based_quantize_dequantize,
    new_dynamic_quantize,
)


class EnhancedActivationExtractor:
    def __init__(self, model_path: str, batch_size: int = 1, max_length: int = 512):
        """
        增强版激活值提取器，捕获更多类型的激活值

        Args:
            model_path: LLaMA模型路径
            batch_size: 批处理大小
            max_length: 最大序列长度
        """
        self.device = torch.device("cpu")
        print(f"使用设备: {self.device}")

        print("正在加载模型...")
        self.model = LlamaForCausalLM.from_pretrained(model_path).to(self.device)
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)

        # 设置padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.handles = []
        self.activations = {}
        self.batch_size = batch_size
        self.max_length = max_length
        print("模型加载完成")

    def _hook_fn(self, name: str):
        """创建钩子函数来捕获激活值"""

        def hook(module, input, output):
            # 如果输出是元组，保存第一个元素
            if isinstance(output, tuple):
                self.activations[name] = output[0].detach().cpu()
            else:
                self.activations[name] = output.detach().cpu()

        return hook

    def _input_hook_fn(self, name: str):
        """创建钩子函数来捕获输入值"""

        def hook(module, input, output):
            # 如果输入是元组，保存第一个元素
            if isinstance(input, tuple) and len(input) > 0:
                self.activations[name] = input[0].detach().cpu()
            else:
                print(f"警告: {name} 的输入不是预期的格式")

        return hook

    def attach_hooks(self, layer_num: int):
        """
        为指定层添加钩子，增加了对更多激活值的捕获

        Args:
            layer_num: 要提取激活值的层号
        """
        try:
            layer = self.model.model.layers[layer_num]

            # 标准激活值钩子
            hook_points = {
                'q_proj': 'q_activation',
                'k_proj': 'k_activation',
                'v_proj': 'v_activation',
                'o_proj': 'o_activation',  # 注意力输出
                'gate_proj': 'ffn_gate_activation',
                'up_proj': 'ffn_up_activation',
                'down_proj': 'ffn_down_activation',
            }

            # 为标准模块添加钩子
            for module_name, activation_name in hook_points.items():
                if hasattr(layer.self_attn, module_name):
                    module = getattr(layer.self_attn, module_name)
                    self.handles.append(
                        module.register_forward_hook(
                            self._hook_fn(f'layer_{layer_num}_{activation_name}')
                        )
                    )
                elif hasattr(layer.mlp, module_name):
                    module = getattr(layer.mlp, module_name)
                    self.handles.append(
                        module.register_forward_hook(
                            self._hook_fn(f'layer_{layer_num}_{activation_name}')
                        )
                    )

            # 捕获Layer Normalization的输入和输出
            # 通常LLaMA有多个LayerNorm
            layernorm_points = {
                'input_layernorm': 'input_ln',
                'post_attention_layernorm': 'post_attn_ln',
            }

            for module_name, activation_name in layernorm_points.items():
                if hasattr(layer, module_name):
                    module = getattr(layer, module_name)
                    # 输入
                    self.handles.append(
                        module.register_forward_hook(
                            self._input_hook_fn(f'layer_{layer_num}_{activation_name}_input')
                        )
                    )
                    # 输出
                    self.handles.append(
                        module.register_forward_hook(
                            self._hook_fn(f'layer_{layer_num}_{activation_name}_output')
                        )
                    )

            # 捕获注意力机制的其他关键部分
            if hasattr(layer, 'self_attn'):
                # 注意力模块整体输出
                self.handles.append(
                    layer.self_attn.register_forward_hook(
                        self._hook_fn(f'layer_{layer_num}_self_attn_output')
                    )
                )

            # 尝试捕获SiLU激活函数后的激活值（LLaMA中的GLU等效部分）
            # 在LLaMA中，GLU由gate_proj和up_proj共同组成，并应用SiLU
            if hasattr(layer, 'mlp'):
                self.handles.append(
                    layer.mlp.register_forward_hook(
                        self._hook_fn(f'layer_{layer_num}_mlp_output')
                    )
                )

            # 尝试捕获RMSNorm的输入和输出（如果存在）
            if hasattr(layer, 'attention_norm'):
                self.handles.append(
                    layer.attention_norm.register_forward_hook(
                        self._input_hook_fn(f'layer_{layer_num}_attention_norm_input')
                    )
                )
                self.handles.append(
                    layer.attention_norm.register_forward_hook(
                        self._hook_fn(f'layer_{layer_num}_attention_norm_output')
                    )
                )

            if hasattr(layer, 'ffn_norm'):
                self.handles.append(
                    layer.ffn_norm.register_forward_hook(
                        self._input_hook_fn(f'layer_{layer_num}_ffn_norm_input')
                    )
                )
                self.handles.append(
                    layer.ffn_norm.register_forward_hook(
                        self._hook_fn(f'layer_{layer_num}_ffn_norm_output')
                    )
                )

            # 对于特定模型结构，尝试捕获sigmoid输出
            # 在LLaMA中，这可能是注意力计算的一部分
            if hasattr(layer.self_attn, 'rotary_emb'):
                self.handles.append(
                    layer.self_attn.rotary_emb.register_forward_hook(
                        self._hook_fn(f'layer_{layer_num}_rotary_emb')
                    )
                )

            # 额外尝试捕获更多可能的激活点
            # 一些特定模型可能有其他独特的模块
            if hasattr(layer.self_attn, 'out_proj'):
                self.handles.append(
                    layer.self_attn.out_proj.register_forward_hook(
                        self._hook_fn(f'layer_{layer_num}_out_proj')
                    )
                )

        except AttributeError as e:
            print(f"添加钩子时出错，模型结构可能与预期不符: {e}")

    def remove_hooks(self):
        """移除所有钩子"""
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def extract_activations(self, layer_num: int, input_text: str) -> Dict[str, torch.Tensor]:
        """
        提取指定层的激活值

        Args:
            layer_num: 要提取的层号
            input_text: 输入文本

        Returns:
            包含该层激活值的字典
        """
        try:
            # 清空之前的激活值
            self.activations = {}

            # 添加钩子
            self.attach_hooks(layer_num)

            # tokenize输入文本
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 打印tokenized长度，用于调试
            input_length = inputs["input_ids"].shape[1]
            print(f"  输入文本长度: {input_length} tokens")

            # 前向传播
            with torch.no_grad():
                self.model(**inputs)

            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return self.activations

        except Exception as e:
            print(f"提取激活值时出错: {e}")
            return {}
        finally:
            self.remove_hooks()


def analyze_activation_distribution(activation: torch.Tensor) -> Dict[str, float]:
    """
    分析激活值分布的统计特性

    Args:
        activation: 激活值张量

    Returns:
        包含分布特性的字典，如偏度、峰度等
    """
    # 展平激活值
    flat_activation = activation.reshape(-1).numpy()

    # 计算基本统计量
    mean = np.mean(flat_activation)
    std = np.std(flat_activation)
    min_val = np.min(flat_activation)
    max_val = np.max(flat_activation)

    # 计算分布形状特性
    skewness = skew(flat_activation)  # 偏度
    kurt = kurtosis(flat_activation)  # 峰度

    # 计算分位数
    q1 = np.percentile(flat_activation, 25)
    q3 = np.percentile(flat_activation, 75)
    iqr = q3 - q1  # 四分位距

    # 计算零值比例和极值比例
    zero_ratio = np.sum(np.abs(flat_activation) < 1e-6) / len(flat_activation)
    extreme_ratio = np.sum(np.abs(flat_activation) > mean + 3 * std) / len(flat_activation)

    # 计算分布不均匀性指标（基尼系数的简化版本）
    sorted_values = np.sort(np.abs(flat_activation))
    cumsum = np.cumsum(sorted_values)
    total = cumsum[-1]
    if total > 0:
        # 计算类似基尼系数的指标
        n = len(sorted_values)
        index_array = np.arange(1, n + 1)
        gini_like = np.sum((2 * index_array - n - 1) * sorted_values) / (n * total)
    else:
        gini_like = 0.0

    return {
        'mean': mean,
        'std': std,
        'min': min_val,
        'max': max_val,
        'range': max_val - min_val,
        'skewness': skewness,  # 偏度，衡量分布的不对称性
        'kurtosis': kurt,  # 峰度，衡量分布尾部的极端值
        'q1': q1,
        'q3': q3,
        'iqr': iqr,
        'zero_ratio': zero_ratio,  # 接近零的值的比例
        'extreme_ratio': extreme_ratio,  # 极端值的比例
        'gini_like': gini_like,  # 分布不均匀性指标
    }


def plot_enhanced_distribution(activation: torch.Tensor,
                               name: str,
                               save_dir: str,
                               stats: Dict[str, float] = None):
    """
    绘制增强版分布图，包括直方图、对数刻度直方图和密度估计

    Args:
        activation: 激活值张量
        name: 激活值名称
        save_dir: 保存目录
        stats: 可选的统计信息字典
    """
    os.makedirs(save_dir, exist_ok=True)

    # 展平激活值
    flat_activation = activation.reshape(-1).numpy()

    # 创建三子图布局
    plt.figure(figsize=(18, 6))

    # 1. 标准直方图
    plt.subplot(1, 3, 1)
    counts, bins, _ = plt.hist(flat_activation, bins=100, color='blue', alpha=0.7)
    plt.title(f"{name} - Linear Scale Distribution")
    plt.xlabel("Activation Value")
    plt.ylabel("Frequency")

    # 2. 对数刻度直方图
    plt.subplot(1, 3, 2)
    plt.hist(flat_activation, bins=100, color='green', alpha=0.7)
    plt.title(f"{name} - Log Scale Distribution")
    plt.xlabel("Activation Value")
    plt.ylabel("Frequency (log scale)")
    plt.yscale('log')

    # 3. 核密度估计
    plt.subplot(1, 3, 3)

    # 使用高斯核密度估计
    # 为了提高效率，对大数据集进行采样
    if len(flat_activation) > 10000:
        sample_indices = np.random.choice(len(flat_activation), 10000, replace=False)
        sample = flat_activation[sample_indices]
    else:
        sample = flat_activation

    # 计算核密度估计
    try:
        # 处理全为常数的情况
        if np.std(sample) < 1e-10:
            x = np.linspace(np.min(sample) - 0.1, np.max(sample) + 0.1, 1000)
            if np.min(sample) == np.max(sample):
                # 如果所有值都相同，绘制一个脉冲
                plt.axvline(x=np.min(sample), color='r')
                plt.text(np.min(sample), 0, f"Constant value: {np.min(sample):.4f}", ha='center')
            else:
                plt.hist(sample, bins=50, density=True, color='red', alpha=0.7)
        else:
            # 确保使用scipy的stats，而不是传入的stats参数
            from scipy import stats as scipy_stats
            kde = scipy_stats.gaussian_kde(sample)
            x = np.linspace(min(sample), max(sample), 1000)
            plt.plot(x, kde(x), 'r-', lw=2)
            plt.fill_between(x, kde(x), alpha=0.3, color='red')
    except Exception as e:
        print(f"核密度估计失败: {e}")
        plt.hist(sample, bins=50, density=True, color='red', alpha=0.7)

    plt.title(f"{name} - Density Estimation")
    plt.xlabel("Activation Value")
    plt.ylabel("Density")

    # 添加统计信息文本框
    if stats:
        info_text = (
            f"Mean: {stats['mean']:.4f}\n"
            f"Std Dev: {stats['std']:.4f}\n"
            f"Min: {stats['min']:.4f}\n"
            f"Max: {stats['max']:.4f}\n"
            f"Skewness: {stats['skewness']:.4f}\n"
            f"Kurtosis: {stats['kurtosis']:.4f}\n"
            f"Zero Ratio: {stats['zero_ratio']:.4f}\n"
            f"Extreme Ratio: {stats['extreme_ratio']:.4f}\n"
            f"Non-uniformity: {stats['gini_like']:.4f}"
        )

        plt.figtext(0.02, 0.02, info_text, fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{name}_distribution.png"))
    plt.close()


def score_distribution_specialness(stats: Dict[str, float]) -> float:
    """
    计算分布特殊程度的分数

    Args:
        stats: 包含分布统计特性的字典

    Returns:
        float: 特殊程度分数，越高表示分布越特殊
    """
    # 定义权重
    weights = {
        'skewness': 1.0,  # 偏度的绝对值越大，分布越不对称
        'kurtosis': 1.0,  # 峰度的绝对值越大，尾部越重或峰越尖
        'zero_ratio': 1.5,  # 零值比例越高，可能是稀疏激活
        'extreme_ratio': 2.0,  # 极端值比例越高，可能有长尾
        'gini_like': 2.0,  # 不均匀性越高，可能是高度集中的分布
    }

    # 计算分数
    score = (
            weights['skewness'] * abs(stats['skewness']) +
            weights['kurtosis'] * abs(stats['kurtosis']) +
            weights['zero_ratio'] * stats['zero_ratio'] +
            weights['extreme_ratio'] * stats['extreme_ratio'] * 10 +  # 乘以10增加极端值的权重
            weights['gini_like'] * stats['gini_like']
    )

    return score


def find_special_distributions(model_path: str,
                               input_texts: List[str],
                               num_layers: int,
                               save_dir: str,
                               top_k: int = None) -> List[Tuple[str, float]]:
    """
    查找模型中分布特殊的激活值

    Args:
        model_path: 模型路径
        input_texts: 输入文本列表
        num_layers: 要分析的层数
        save_dir: 保存目录
        top_k: 返回的最特殊分布数量

    Returns:
        特殊分布列表，每项包含名称和特殊程度分数
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 创建激活值提取器
    extractor = EnhancedActivationExtractor(model_path)

    # 用于存储所有特殊分布的列表
    all_distributions = []

    # 使用单个输入文本单独处理每个分布
    print("使用单个输入模式进行分析...")

    # 遍历所有层
    for layer_num in tqdm(range(num_layers), desc="分析层"):
        print(f"\n分析第 {layer_num} 层...")

        # 使用不同输入文本提取激活值并分别处理
        all_layer_activations = {}

        for i, text in enumerate(input_texts):
            print(f"  使用输入文本 {i + 1}/{len(input_texts)}")

            try:
                activations = extractor.extract_activations(layer_num, text)

                # 处理每个激活值
                for name, activation in activations.items():
                    # 记录形状信息
                    shape = activation.shape
                    print(f"    {name}: 形状 = {shape}")

                    # 计算统计特性
                    stats = analyze_activation_distribution(activation)

                    # 计算特殊程度分数
                    specialness_score = score_distribution_specialness(stats)

                    # 生成唯一的标识符，包含输入索引
                    unique_name = f"{name}_input{i}"
                    all_layer_activations[unique_name] = (activation, specialness_score, stats)

                    print(f"    {unique_name}: 特殊程度分数 = {specialness_score:.4f}")

            except Exception as e:
                print(f"  处理输入 {i + 1} 时出错: {e}")

        # 添加所有处理过的激活值
        for unique_name, (activation, score, stats) in all_layer_activations.items():
            all_distributions.append((unique_name, activation, score, stats))

    # 按特殊程度分数排序
    all_distributions.sort(key=lambda x: x[2], reverse=True)

    # 输出并保存所有分布（按特殊程度排序）
    print("\n激活值分布（按特殊程度排序）:")
    special_distributions = []

    # 如果top_k为None，处理所有分布；否则只处理前top_k个
    distributions_to_process = all_distributions[:top_k] if top_k is not None else all_distributions

    for i, (name, activation, score, stats) in enumerate(distributions_to_process):
        print(f"{i + 1}. {name}: 特殊程度分数 = {score:.4f}")
        special_distributions.append((name, score))

        # 创建分布可视化
        safe_name = name.replace("/", "_").replace(".", "_")
        plot_enhanced_distribution(activation, safe_name, save_dir, stats)

        # 记录统计信息
        stats_file = os.path.join(save_dir, f"{safe_name}_stats.txt")
        with open(stats_file, "w") as f:
            f.write(f"激活值: {name}\n")
            f.write(f"特殊程度分数: {score:.6f}\n\n")
            f.write("统计信息:\n")
            for stat_name, stat_value in stats.items():
                f.write(f"{stat_name}: {stat_value:.6f}\n")

    return special_distributions


def compare_quantization_methods_for_special_distributions(model_path: str,
                                                           special_distributions: List[Tuple[str, float]],
                                                           input_text: str,
                                                           save_dir: str,
                                                           all_input_texts: List[str] = None):
    """
    对特殊分布的激活值比较动态量化和均匀量化的效果

    Args:
        model_path: 模型路径
        special_distributions: 特殊分布列表
        input_text: 输入文本
        save_dir: 保存目录
        all_input_texts: 所有输入文本列表，用于原始分析中使用的文本
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 创建激活值提取器
    extractor = EnhancedActivationExtractor(model_path)

    # 记录比较结果
    results_file = os.path.join(save_dir, "quantization_comparison.txt")
    with open(results_file, "w") as f:
        f.write("特殊分布激活值量化比较\n")
        f.write("=" * 50 + "\n\n")

    # 跟踪最佳改进
    best_improvements = []

    # 如果没有提供all_input_texts，创建一个默认列表
    if all_input_texts is None:
        all_input_texts = [input_text]

    # 遍历特殊分布
    for i, (name, score) in enumerate(tqdm(special_distributions, desc="量化特殊分布")):
        # 提取基本信息
        print(f"\n处理分布 {i + 1}/{len(special_distributions)}: {name}")

        # 修复输入索引提取问题
        # 使用正则表达式提取输入索引
        input_idx_match = re.search(r'_input(\d+)$', name)
        if input_idx_match:
            input_idx = int(input_idx_match.group(1))
            original_name = name[:name.rfind('_input')]
        else:
            input_idx = 0
            original_name = name

        # 解析层号
        layer_parts = original_name.split('_')
        if len(layer_parts) > 1 and layer_parts[0] == 'layer' and layer_parts[1].isdigit():
            layer_num = int(layer_parts[1])
        else:
            print(f"  无法解析层号，跳过: {name}")
            continue

        # 使用正确的输入文本
        current_input = all_input_texts[input_idx] if input_idx < len(all_input_texts) else input_text

        # 提取激活值
        print(f"  提取层 {layer_num} 的激活值，使用输入 {input_idx}")
        activations = extractor.extract_activations(layer_num, current_input)

        # 检查原始激活值名称是否存在
        if original_name in activations:
            activation = activations[original_name]
            safe_name = name.replace("/", "_").replace(".", "_")

            print(f"  激活值形状: {activation.shape}")

            try:
                # 量化并评估
                final_dequantized, hist_data, quantized_data, scale, bias = new_dynamic_quantize(
                    data=activation,
                    initial_levels=16,  # 2^4
                    target_levels=8,  # 2^3
                    verbose=False
                )

                # 均匀量化对比
                uniform_quantized, uniform_dequantized = interval_based_quantize_dequantize(
                    data=activation,
                    num_levels=8
                )

                # 计算误差
                dynamic_error = ((final_dequantized - activation) ** 2).mean().item()
                uniform_error = ((uniform_dequantized - activation) ** 2).mean().item()

                # 计算改进百分比
                improvement = ((uniform_error - dynamic_error) / uniform_error * 100)

                # 跟踪最佳改进
                best_improvements.append((name, improvement, dynamic_error, uniform_error))

                # 保存量化结果可视化
                plt.figure(figsize=(20, 10))

                # 原始分布
                plt.subplot(2, 2, 1)
                plt.hist(activation.reshape(-1).numpy(), bins=100, color='blue', alpha=0.7)
                plt.title("Original Activation Distribution")
                plt.xlabel("Value")
                plt.ylabel("Frequency")

                # 对数刻度
                plt.subplot(2, 2, 2)
                plt.hist(activation.reshape(-1).numpy(), bins=100, color='blue', alpha=0.7)
                plt.title("Original Activation Distribution (Log Scale)")
                plt.xlabel("Value")
                plt.ylabel("Frequency (Log Scale)")
                plt.yscale('log')

                # 动态量化结果
                plt.subplot(2, 2, 3)
                plt.hist(final_dequantized.reshape(-1).numpy(), bins=100, color='red', alpha=0.7)
                plt.title(f"Dynamic Quantization (8 levels) - MSE: {dynamic_error:.6f}")
                plt.xlabel("Value")
                plt.ylabel("Frequency (Log Scale)")
                plt.yscale('log')

                # 均匀量化结果
                plt.subplot(2, 2, 4)
                plt.hist(uniform_dequantized.reshape(-1).numpy(), bins=100, color='green', alpha=0.7)
                plt.title(f"Uniform Quantization (8 levels) - MSE: {uniform_error:.6f}")
                plt.xlabel("Value")
                plt.ylabel("Frequency (Log Scale)")
                plt.yscale('log')

                # 添加总体结果文本
                plt.figtext(0.5, 0.01,
                            f"Dynamic vs Uniform Improvement: {improvement:.2f}%",
                            ha="center", fontsize=12,
                            bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5})

                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"{safe_name}_quantization_comparison.png"))
                plt.close()

                # 记录结果
                with open(results_file, "a") as f:
                    f.write(f"激活值: {name}\n")
                    f.write(f"动态量化MSE: {dynamic_error:.6f}\n")
                    f.write(f"均匀量化MSE: {uniform_error:.6f}\n")
                    f.write(f"改进百分比: {improvement:.2f}%\n")
                    f.write("-" * 50 + "\n\n")

                print(f"  量化比较结果:")
                print(f"  动态量化MSE: {dynamic_error:.6f}")
                print(f"  均匀量化MSE: {uniform_error:.6f}")
                print(f"  改进百分比: {improvement:.2f}%")

            except Exception as e:
                print(f"  量化过程中出错: {e}")
        else:
            print(f"  警告: 未找到激活值 {original_name}，可用的激活值: {list(activations.keys())}")

    # 排序并保存最佳改进结果
    if best_improvements:
        best_improvements.sort(key=lambda x: x[1], reverse=True)

        with open(os.path.join(save_dir, "best_improvements.txt"), "w") as f:
            f.write("动态量化最佳改进排名\n")
            f.write("=" * 50 + "\n\n")

            for i, (name, improvement, dynamic_error, uniform_error) in enumerate(best_improvements[:50]):
                f.write(f"{i + 1}. {name}\n")
                f.write(f"   动态量化MSE: {dynamic_error:.6f}\n")
                f.write(f"   均匀量化MSE: {uniform_error:.6f}\n")
                f.write(f"   改进百分比: {improvement:.2f}%\n")
                f.write("-" * 50 + "\n\n")

        print("\n动态量化最佳改进排名 (前10):")
        for i, (name, improvement, dynamic_error, uniform_error) in enumerate(best_improvements[:10]):
            print(f"{i + 1}. {name}: 改进 {improvement:.2f}%")


def main():
    # 配置参数
    model_path = "/home/yuzhezhang/.llama/checkpoints/Llama-2-7b-hf"
    save_base_dir = "special_activation_distributions"

    # 原始的10个测试句子
    input_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "I believe artificial intelligence will transform the world in many unexpected ways.",
        "The history of machine learning can be traced back to early statistical methods.",
        "Language models have demonstrated remarkable abilities to understand and generate human language.",
        "Quantum computing promises to solve problems that classical computers cannot efficiently address.",
        "The development of large language models has accelerated in recent years.",
        "Transfer learning has revolutionized how we approach natural language processing tasks.",
        "Neural networks can approximate any continuous function given enough parameters.",
        "The transformer architecture has become the dominant approach in modern NLP systems.",
        "Attention mechanisms allow models to focus on relevant parts of the input sequence."
    ]

    # 使用更长、更复杂的输入文本进行量化比较
    main_input_text = "The performance of large language models depends on their architecture, training data quality, and optimization techniques. These models have demonstrated impressive capabilities in understanding context and generating coherent responses."

    try:
        # 1. 找出特殊分布（分析所有32层）
        special_distributions = find_special_distributions(
            model_path=model_path,
            input_texts=input_texts,
            num_layers=32,  # 分析所有32层
            save_dir=f"{save_base_dir}/distributions",
            top_k=None  # 返回所有特殊分布
        )

        # 2. 比较量化方法（传递input_texts列表）
        # 对于完整分析，可以限制只对前100个最特殊的分布进行量化分析，以节省时间
        top_special_distributions = special_distributions[:100] if len(
            special_distributions) > 100 else special_distributions

        print(f"\n找到 {len(special_distributions)} 个特殊分布，对前 {len(top_special_distributions)} 个进行量化分析")

        compare_quantization_methods_for_special_distributions(
            model_path=model_path,
            special_distributions=top_special_distributions,
            input_text=main_input_text,
            save_dir=f"{save_base_dir}/quantization",
            all_input_texts=input_texts  # 传入完整的输入文本列表
        )

        print("\n分析完成！")
        print(f"分布可视化保存在: {save_base_dir}/distributions")
        print(f"量化比较结果保存在: {save_base_dir}/quantization")

    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()