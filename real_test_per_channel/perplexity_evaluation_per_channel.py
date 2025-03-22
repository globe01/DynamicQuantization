import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Tuple
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    TextDataset,
    DataCollatorForLanguageModeling
)
from torch.utils.data import DataLoader


# from real_quantization_utils import (
#     interval_based_quantize_dequantize,
#     new_dynamic_quantize,
# )


# 新增：逐通道量化函数修改
def interval_based_quantize_dequantize_per_channel(data, num_levels, dequantize=True):
    """
    基于均匀区间的逐通道量化函数，支持反量化。

    参数:
        data: torch.Tensor，输入的浮点数据，形状为 [out_features, in_features]
        num_levels: int，量化级别数量
        dequantize: bool，是否进行反量化，默认为 True

    返回:
        quantized_data: torch.Tensor，量化后的数据
        dequantized_data: torch.Tensor，反量化后的数据（如果 dequantize=True）
    """
    # 获取输入张量的形状
    shape = data.shape
    out_features = shape[0]

    # 创建输出张量
    quantized_data = torch.zeros_like(data)
    dequantized_data = torch.zeros_like(data)

    # 对每个输出通道单独量化
    for i in range(out_features):
        # 获取当前通道的权重
        channel_data = data[i]

        # 获取通道数据的最小值和最大值
        min_val = torch.min(channel_data)
        max_val = torch.max(channel_data)

        # 计算区间的步长
        interval_size = (max_val - min_val) / num_levels

        # 对于只有一个值的通道特殊处理
        if min_val == max_val:
            quantized_data[i] = torch.zeros_like(channel_data)
            dequantized_data[i] = channel_data
            continue

        # 量化当前通道
        channel_quantized = torch.round((channel_data - min_val - interval_size / 2) / interval_size)
        channel_quantized = torch.clamp(channel_quantized, 0, num_levels - 1)

        # 反量化当前通道
        channel_dequantized = channel_quantized * interval_size + min_val + interval_size / 2

        # 保存结果
        quantized_data[i] = channel_quantized
        dequantized_data[i] = channel_dequantized

    if dequantize:
        return quantized_data, dequantized_data
    else:
        return quantized_data


# 新增：动态量化函数修改
def new_dynamic_quantize_per_channel(data, initial_levels, target_levels, verbose=True):
    """
    新的逐通道动态量化函数，对每个输出通道单独进行动态量化。

    参数：
        data: torch.Tensor, 需要量化的输入数据，形状为 [out_features, in_features]
        initial_levels: int, 初始量化级别数
        target_levels: int, 动态量化后的目标级别数
        verbose: bool, 是否打印详细信息
    """
    # 获取输入张量的形状
    shape = data.shape
    out_features = shape[0]

    # 创建输出张量
    final_dequantized = torch.zeros_like(data)

    # 记录总体量化误差
    total_error = 0.0

    # 对每个输出通道单独量化
    for i in range(out_features):
        # 获取当前通道的权重
        channel_data = data[i]

        # 步骤1：计算缩放参数
        min_val = torch.min(channel_data)
        max_val = torch.max(channel_data)

        # 对于只有一个值的通道特殊处理
        if min_val == max_val:
            final_dequantized[i] = channel_data
            continue

        interval_size = (max_val - min_val) / initial_levels

        if verbose and i == 0:
            print(f"\n通道 {i} 初始均匀量化:")
            print(f"数据范围: [{min_val:.4f}, {max_val:.4f}]")
            print(f"量化区间大小: {interval_size:.4f}")

        # 对原始数据进行相同的平移缩放操作(但不量化)
        scaled_data = (channel_data - min_val - interval_size / 2) / interval_size

        # 执行初始量化
        quantized_data = torch.round(scaled_data)  # 四舍五入
        quantized_data = torch.clamp(quantized_data, 0, initial_levels - 1)  # 限制范围

        if verbose and i == 0:
            print(f"初始量化后的级别数: {len(torch.unique(quantized_data))}")

        # 步骤2：动态量化
        unique_levels = torch.unique(quantized_data)
        unique_levels.sort()

        # 跳过已经满足目标级别的通道
        if len(unique_levels) <= target_levels:
            # 直接反量化
            scale = interval_size
            bias = min_val + interval_size / 2
            final_dequantized[i] = quantized_data * scale + bias
            continue

        iteration = 0
        hist_data = []

        # 初始化level_mapping字典
        level_mapping = {}
        for level in unique_levels:
            level_mapping[float(level.item())] = float(level.item())

        if verbose and i == 0:
            print("\n开始动态量化过程...")

        while len(unique_levels) > target_levels:
            iteration += 1
            if verbose and i == 0 and iteration <= 3:  # 只打印前3次迭代
                print(f"\n迭代 {iteration}:")
                print(f"当前量化级别: {unique_levels.tolist()}")
                print(f"剩余级别数: {len(unique_levels)}")

            total_errors = {}  # 存储移除每个级别后的误差

            # 尝试移除每个级别并计算产生的误差
            for level in unique_levels:
                level = float(level.item())
                Q_copy = quantized_data.clone()
                level_idx = torch.where(unique_levels == level)[0].item()

                if level_idx == 0:  # 第一个级别
                    replacement = unique_levels[1]
                elif level_idx == len(unique_levels) - 1:  # 最后一个级别
                    replacement = unique_levels[-2]
                else:  # 中间级别
                    less_level = float(unique_levels[level_idx - 1].item())
                    more_level = float(unique_levels[level_idx + 1].item())

                    # 基于反量化值的距离计算
                    less_level_value = level_mapping[less_level]
                    more_level_value = level_mapping[more_level]
                    min_level_value = level_mapping[level]

                    replacement = (
                        less_level
                        if abs(min_level_value - less_level_value) < abs(min_level_value - more_level_value)
                        else more_level
                    )

                Q_copy[Q_copy == level] = replacement

                # 使用scaled_data计算误差
                total_error = ((Q_copy - scaled_data) ** 2).sum().item()
                total_errors[level] = total_error

            # 找到移除时产生最小误差的级别
            level_to_remove = min(total_errors, key=total_errors.get)
            level_idx = torch.where(unique_levels == level_to_remove)[0].item()

            if verbose and i == 0 and iteration <= 3:  # 只打印前3次迭代
                print(f"选择移除级别: {level_to_remove}")

            # 移除选定的级别
            if level_idx == 0:
                replacement = unique_levels[1]
            elif level_idx == len(unique_levels) - 1:
                replacement = unique_levels[-2]
            else:
                less_level = float(unique_levels[level_idx - 1].item())
                more_level = float(unique_levels[level_idx + 1].item())

                less_level_value = level_mapping[less_level]
                more_level_value = level_mapping[more_level]
                level_to_remove_value = level_mapping[level_to_remove]

                replacement = (
                    less_level
                    if abs(level_to_remove_value - less_level_value) < abs(level_to_remove_value - more_level_value)
                    else more_level
                )

            # 更新量化数据
            quantized_data[quantized_data == level_to_remove] = replacement

            # 更新level_mapping
            if float(level_to_remove) in level_mapping:
                replacement_value = float(replacement)
                level_mapping[replacement_value] = (level_mapping[replacement_value] + level_mapping[
                    float(level_to_remove)]) / 2
                del level_mapping[float(level_to_remove)]

            unique_levels = torch.unique(quantized_data)
            unique_levels.sort()
            hist_data.append(quantized_data.clone().numpy())

        # 步骤3：最终反量化
        scale = interval_size
        bias = min_val + interval_size / 2
        channel_dequantized = quantized_data * scale + bias

        # 计算通道量化误差
        channel_error = ((channel_dequantized - channel_data) ** 2).mean().item()
        total_error += channel_error

        # 存储反量化结果
        final_dequantized[i] = channel_dequantized

        if verbose and i == 0:
            print("\n量化过程完成:")
            print(f"最终量化级别: {unique_levels.tolist()}")
            print(f"最终级别数: {len(unique_levels)}")
            print(f"通道量化误差: {channel_error:.6f}")

    # 计算平均量化误差
    avg_error = total_error / out_features

    if verbose:
        print(f"\n所有通道量化完成:")
        print(f"平均量化误差: {avg_error:.6f}")

    return final_dequantized, hist_data, avg_error


# 设置设备
device = torch.device("cpu")
print(f"使用设备: {device}")


class ModelQuantizer:
    """模型量化器，负责对模型权重进行量化和替换"""

    def __init__(self, model_path: str):
        """
        初始化模型量化器

        Args:
            model_path: 模型路径
        """
        self.model_path = model_path
        self.model = LlamaForCausalLM.from_pretrained(model_path)
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)

        # 确保模型处于评估模式
        self.model.eval()

        # 保存原始权重的副本
        self.original_weights = {}
        self.save_original_weights()

        # 跟踪量化的层
        self.quantized_layers = set()

        # 获取应该被量化的权重名称列表（只包含变压器中的矩阵乘法权重）
        self.quantizable_weights = self.get_quantizable_weights()
        print(f"找到 {len(self.quantizable_weights)} 个可量化的权重参数")

    def get_quantizable_weights(self):
        """获取应该被量化的权重名称列表（只包含变压器中的矩阵乘法权重）"""
        quantizable_weights = []

        for name, param in self.model.named_parameters():
            # 只量化以下权重:
            # 1. 注意力层 的查询、键、值和输出矩阵
            # 2. 前馈网络层 的矩阵
            if any(pattern in name for pattern in [
                ".self_attn.q_proj.weight",
                ".self_attn.k_proj.weight",
                ".self_attn.v_proj.weight",
                ".self_attn.o_proj.weight",
                ".mlp.gate_proj.weight",
                ".mlp.up_proj.weight",
                ".mlp.down_proj.weight"
            ]):
                quantizable_weights.append(name)

        return quantizable_weights

    def save_original_weights(self):
        """保存模型的原始权重"""
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.requires_grad:
                self.original_weights[name] = param.data.clone()

        print(f"已保存 {len(self.original_weights)} 个原始权重参数")

    def restore_original_weights(self):
        """恢复模型的原始权重"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.original_weights:
                    param.data = self.original_weights[name].clone()

        # 清空已量化层记录
        self.quantized_layers = set()
        print("已恢复所有原始权重")

    def apply_dynamic_quantization(self, initial_levels=32, target_levels=16):
        """
        对模型应用逐通道动态量化

        Args:
            initial_levels: 初始量化级别
            target_levels: 目标量化级别
        """
        print(f"\n开始应用逐通道动态量化 (从 {initial_levels} 级别 到 {target_levels} 级别)...")
        self.restore_original_weights()  # 先恢复原始权重

        quantized_count = 0
        total_dynamic_error = 0.0
        with torch.no_grad():
            for name, param in tqdm(list(self.model.named_parameters())):
                # 只量化特定的权重矩阵
                if name in self.quantizable_weights:
                    # 应用动态量化
                    weight_tensor = param.data.cpu()

                    # 打印量化前的权重统计信息
                    if quantized_count == 0:  # 只打印第一个层的详细信息
                        print(f"\n量化层: {name}")
                        print(f"量化前权重统计: 最小值={weight_tensor.min().item():.6f}, "
                              f"最大值={weight_tensor.max().item():.6f}, "
                              f"均值={weight_tensor.mean().item():.6f}, "
                              f"标准差={weight_tensor.std().item():.6f}")

                    # 应用逐通道动态量化
                    final_dequantized, _, avg_error = new_dynamic_quantize_per_channel(
                        data=weight_tensor,
                        initial_levels=initial_levels,
                        target_levels=target_levels,
                        verbose=(quantized_count == 0)  # 只对第一层详细打印
                    )

                    # 检查量化结果是否包含NaN或无穷大
                    if torch.isnan(final_dequantized).any() or torch.isinf(final_dequantized).any():
                        print(f"警告: 层 {name} 的量化结果包含NaN或无穷大值!")
                        continue

                    # 累加量化误差
                    total_dynamic_error += avg_error

                    # 打印量化后的权重统计信息
                    if quantized_count == 0:  # 只打印第一个层的详细信息
                        print(f"量化后权重统计: 最小值={final_dequantized.min().item():.6f}, "
                              f"最大值={final_dequantized.max().item():.6f}, "
                              f"均值={final_dequantized.mean().item():.6f}, "
                              f"标准差={final_dequantized.std().item():.6f}")
                        print(f"量化误差: {avg_error:.6f}")

                    # 更新权重
                    param.data = final_dequantized.to(param.device)
                    self.quantized_layers.add(name)
                    quantized_count += 1

        avg_dynamic_error = total_dynamic_error / quantized_count if quantized_count > 0 else 0
        print(f"已对 {quantized_count} 个层应用逐通道动态量化")
        print(f"平均动态量化误差: {avg_dynamic_error:.6f}")
        return avg_dynamic_error

    def apply_uniform_quantization(self, target_levels=16):
        """
        对模型应用逐通道均匀量化

        Args:
            target_levels: 量化级别
        """
        print(f"\n开始应用逐通道均匀量化 (到 {target_levels} 级别)...")
        self.restore_original_weights()  # 先恢复原始权重

        quantized_count = 0
        total_uniform_error = 0.0
        with torch.no_grad():
            for name, param in tqdm(list(self.model.named_parameters())):
                # 只量化特定的权重矩阵
                if name in self.quantizable_weights:
                    # 应用均匀量化
                    weight_tensor = param.data.cpu()

                    # 打印量化前的权重统计信息
                    if quantized_count == 0:  # 只打印第一个层的详细信息
                        print(f"\n量化层: {name}")
                        print(f"量化前权重统计: 最小值={weight_tensor.min().item():.6f}, "
                              f"最大值={weight_tensor.max().item():.6f}, "
                              f"均值={weight_tensor.mean().item():.6f}, "
                              f"标准差={weight_tensor.std().item():.6f}")

                    # 应用逐通道均匀量化
                    _, uniform_dequantized = interval_based_quantize_dequantize_per_channel(
                        data=weight_tensor,
                        num_levels=target_levels
                    )

                    # 检查量化结果是否包含NaN或无穷大
                    if torch.isnan(uniform_dequantized).any() or torch.isinf(uniform_dequantized).any():
                        print(f"警告: 层 {name} 的量化结果包含NaN或无穷大值!")
                        continue

                    # 计算量化误差
                    layer_error = ((uniform_dequantized - weight_tensor) ** 2).mean().sqrt().item()
                    total_uniform_error += layer_error

                    # 打印量化后的权重统计信息
                    if quantized_count == 0:  # 只打印第一个层的详细信息
                        print(f"量化后权重统计: 最小值={uniform_dequantized.min().item():.6f}, "
                              f"最大值={uniform_dequantized.max().item():.6f}, "
                              f"均值={uniform_dequantized.mean().item():.6f}, "
                              f"标准差={uniform_dequantized.std().item():.6f}")
                        print(f"量化误差: {layer_error:.6f}")

                    # 更新权重
                    param.data = uniform_dequantized.to(param.device)
                    self.quantized_layers.add(name)
                    quantized_count += 1

        avg_uniform_error = total_uniform_error / quantized_count if quantized_count > 0 else 0
        print(f"已对 {quantized_count} 个层应用逐通道均匀量化")
        print(f"平均均匀量化误差: {avg_uniform_error:.6f}")
        return avg_uniform_error

    def get_model(self):
        """获取当前模型"""
        return self.model

    def get_tokenizer(self):
        """获取分词器"""
        return self.tokenizer


class PerplexityEvaluator:
    """困惑度评估器，负责计算模型在给定数据集上的困惑度"""

    def __init__(self, model, tokenizer, dataset_path=None, max_length=512):
        """
        初始化困惑度评估器

        Args:
            model: 要评估的模型
            tokenizer: 分词器
            dataset_path: 数据集路径，如果为None则使用默认测试句子
            max_length: 最大序列长度
        """
        self.model = model
        self.tokenizer = tokenizer
        self.dataset_path = dataset_path
        self.max_length = max_length

        # 设置padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 如果没有提供数据集，使用默认测试句子
        self.default_test_sentences = [
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

    def prepare_dataloader(self, batch_size=2):
        """
        准备数据加载器

        Args:
            batch_size: 批处理大小

        Returns:
            数据加载器
        """
        if self.dataset_path and os.path.exists(self.dataset_path):
            # 使用自定义数据集
            dataset = TextDataset(
                tokenizer=self.tokenizer,
                file_path=self.dataset_path,
                block_size=self.max_length
            )

            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )

            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=data_collator
            )

            return dataloader
        else:
            # 使用默认测试句子
            print("未找到数据集，使用默认测试句子")
            return None

    def calculate_perplexity_on_sentences(self, sentences=None):
        """
        在给定句子上计算困惑度，优化内存使用

        Args:
            sentences: 要评估的句子列表，如果为None则使用默认测试句子

        Returns:
            平均困惑度
        """
        if sentences is None:
            sentences = self.default_test_sentences

        # 提前将模型切换到评估模式
        self.model.eval()

        total_loss = 0
        total_tokens = 0
        has_nan_inf = False

        print("\n" + "=" * 50)
        print("开始计算困惑度评估...")

        # 每次处理一个句子，减少内存使用
        with torch.no_grad():
            for i, sentence in enumerate(tqdm(sentences, desc="计算困惑度")):
                try:
                    # 每次处理前清理CUDA缓存
                    torch.cuda.empty_cache()

                    # 只将当前需要的部分转移到GPU
                    self.model.to(device)

                    # 处理输入
                    inputs = self.tokenizer(sentence, return_tensors="pt")
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    # 计算损失
                    outputs = self.model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss

                    # 记录损失
                    if not torch.isnan(loss) and not torch.isinf(loss):
                        total_loss += loss.item() * inputs["input_ids"].size(1)
                        total_tokens += inputs["input_ids"].size(1)
                    else:
                        print(f"警告: 句子 {i + 1} 产生了NaN或Inf损失，将跳过")
                        has_nan_inf = True

                    # 在每次迭代后立即将模型移回CPU并清理显存
                    self.model.to('cpu')
                    del inputs, outputs, loss
                    torch.cuda.empty_cache()

                except Exception as e:
                    print(f"处理句子 {i + 1} 时出错: {e}")
                    # 确保模型回到CPU
                    self.model.to('cpu')
                    torch.cuda.empty_cache()
                    continue

        # 检查是否有有效的损失值
        if total_tokens == 0:
            print("警告: 没有有效的token或所有句子都产生了无效结果")
            return 1000.0  # 返回最大困惑度

        # 计算平均困惑度
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss))

        print(f"\n评估完成: 总token数: {total_tokens}, 平均损失: {avg_loss:.4f}, 困惑度: {perplexity.item():.4f}")
        print("=" * 50)

        # 如果困惑度超过1000或异常，进行警告
        if perplexity.item() > 1000 or torch.isnan(perplexity) or torch.isinf(perplexity):
            print(f"警告: 困惑度异常高 ({perplexity.item()})，可能表明量化导致了严重的模型退化")
            return min(1000.0, perplexity.item()) if not torch.isnan(perplexity) and not torch.isinf(
                perplexity) else 1000.0

        return perplexity.item()

    def calculate_perplexity_on_dataset(self, dataloader):
        """
        在给定数据集上计算困惑度

        Args:
            dataloader: 数据加载器

        Returns:
            平均困惑度
        """
        self.model.to(device)
        self.model.eval()

        total_loss = 0
        total_tokens = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="计算困惑度"):
                inputs = {k: v.to(device) for k, v in batch.items()}

                # 计算损失
                outputs = self.model(**inputs)
                loss = outputs.loss

                # 累加损失和token数量
                total_loss += loss.item() * inputs["input_ids"].size(0) * inputs["input_ids"].size(1)
                total_tokens += inputs["input_ids"].size(0) * inputs["input_ids"].size(1)

        # 计算平均困惑度
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss))

        return perplexity.item()

    def evaluate(self):
        """
        评估模型的困惑度

        Returns:
            困惑度分数
        """
        dataloader = self.prepare_dataloader()

        if dataloader:
            return self.calculate_perplexity_on_dataset(dataloader)
        else:
            return self.calculate_perplexity_on_sentences()


def compare_quantization_methods(model_path, results_dir="quantization_perplexity_results"):
    """
    比较不同量化方法的困惑度

    Args:
        model_path: 模型路径
        results_dir: 结果保存目录
    """
    os.makedirs(results_dir, exist_ok=True)

    # 创建模型量化器
    quantizer = ModelQuantizer(model_path)

    # 测试的量化级别配置
    test_configurations = [
        {"name": "Original (FP32)", "method": "none", "init_levels": None, "target_levels": None},
        {"name": "Dynamic (32→16)", "method": "dynamic", "init_levels": 32, "target_levels": 16},
        {"name": "Uniform (16)", "method": "uniform", "init_levels": None, "target_levels": 16}
    ]

    results = []
    quantization_errors = {"dynamic": 0.0, "uniform": 0.0}

    # 对每种配置进行测试
    for config in test_configurations:
        print(f"\n{'=' * 60}")
        print(f"测试配置: {config['name']}")
        print(f"{'=' * 60}")

        # 应用相应的量化方法
        if config["method"] == "none":
            quantizer.restore_original_weights()
            print("使用原始权重 (FP32)")
        elif config["method"] == "dynamic":
            error = quantizer.apply_dynamic_quantization(
                initial_levels=config["init_levels"],
                target_levels=config["target_levels"]
            )
            quantization_errors["dynamic"] = error
        elif config["method"] == "uniform":
            error = quantizer.apply_uniform_quantization(
                target_levels=config["target_levels"]
            )
            quantization_errors["uniform"] = error

        # 创建评估器并计算困惑度
        evaluator = PerplexityEvaluator(
            model=quantizer.get_model(),
            tokenizer=quantizer.get_tokenizer()
        )

        start_time = time.time()
        try:
            perplexity = evaluator.evaluate()
            # 限制困惑度的上限，防止图表显示问题
            if perplexity > 1000 or np.isinf(perplexity):
                print(f"警告: 困惑度值异常大 ({perplexity})，将被限制为1000")
                perplexity = 1000
        except Exception as e:
            print(f"评估时出错: {e}")
            perplexity = 1000  # 出错时设置为一个大值
        end_time = time.time()

        # 记录结果
        result = {
            "name": config["name"],
            "method": config["method"],
            "perplexity": perplexity,
            "evaluation_time": end_time - start_time
        }

        # 添加量化误差信息（如果适用）
        if config["method"] in quantization_errors:
            result["quantization_error"] = quantization_errors[config["method"]]

        results.append(result)

        print(f"配置: {config['name']}")
        print(f"困惑度: {perplexity:.4f}")
        print(f"评估时间: {end_time - start_time:.2f} 秒")

    # 保存和可视化结果
    save_and_visualize_results(results, results_dir)

    return results


def save_and_visualize_results(results, results_dir):
    """
    保存和可视化结果

    Args:
        results: 结果列表
        results_dir: 结果保存目录
    """
    # 保存为文本文件
    with open(os.path.join(results_dir, "perplexity_results.txt"), "w") as f:
        f.write("Configuration\tPerplexity\tEvaluation Time (s)\tQuantization Error\n")
        for result in results:
            quant_error = result.get("quantization_error", "N/A")
            f.write(f"{result['name']}\t{result['perplexity']:.4f}\t{result['evaluation_time']:.2f}\t{quant_error}\n")

    # 绘制困惑度对比图
    plt.figure(figsize=(12, 6))

    # 提取数据
    names = [result["name"] for result in results]
    perplexities = [result["perplexity"] for result in results]

    # 使用不同颜色区分不同量化方法
    colors = ['blue', 'green', 'red'][:len(results)]

    # 创建柱状图
    bars = plt.bar(names, perplexities, color=colors)

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom', fontsize=10)

    # 设置图表属性
    plt.title("Perplexity Comparison of Different Quantization Methods")
    plt.xlabel("Quantization Method")
    plt.ylabel("Perplexity (lower is better)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # 保存图表
    plt.savefig(os.path.join(results_dir, "perplexity_comparison.png"), dpi=300)
    plt.close()

    # 绘制评估时间对比图
    plt.figure(figsize=(12, 6))

    # 提取评估时间数据
    times = [result["evaluation_time"] for result in results]

    # 创建柱状图
    bars = plt.bar(names, times, color=colors)

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.2f}s',
                 ha='center', va='bottom', fontsize=10)

    # 设置图表属性
    plt.title("Evaluation Time Comparison of Different Quantization Methods")
    plt.xlabel("Quantization Method")
    plt.ylabel("Evaluation Time (seconds)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # 保存图表
    plt.savefig(os.path.join(results_dir, "evaluation_time_comparison.png"), dpi=300)
    plt.close()

    # 绘制量化误差对比图（如果有）
    quant_errors = [result.get("quantization_error", 0) for result in results if "quantization_error" in result]
    quant_methods = [result["name"] for result in results if "quantization_error" in result]

    if quant_errors and quant_methods:
        plt.figure(figsize=(12, 6))

        # 创建柱状图
        bars = plt.bar(quant_methods, quant_errors, color=colors[1:])

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.6f}',
                     ha='center', va='bottom', fontsize=10)

        # 设置图表属性
        plt.title("Quantization Error Comparison")
        plt.xlabel("Quantization Method")
        plt.ylabel("Mean Squared Error")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # 保存图表
        plt.savefig(os.path.join(results_dir, "quantization_error_comparison.png"), dpi=300)
        plt.close()

    # 绘制性能折衷图（困惑度 vs 量化误差）
    if quant_errors and quant_methods:
        quant_perplexities = [result["perplexity"] for result in results if "quantization_error" in result]

        plt.figure(figsize=(12, 6))

        # 创建散点图
        plt.scatter(quant_errors, quant_perplexities, color=colors[1:], s=100)

        # 添加标签
        for i, method in enumerate(quant_methods):
            plt.annotate(method, (quant_errors[i], quant_perplexities[i]),
                         xytext=(10, 5), textcoords='offset points')

        # 设置图表属性
        plt.title("Perplexity vs Quantization Error")
        plt.xlabel("Quantization Error (MSE)")
        plt.ylabel("Perplexity (lower is better)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        # 保存图表
        plt.savefig(os.path.join(results_dir, "perplexity_vs_error.png"), dpi=300)
        plt.close()

    print(f"结果已保存到 {results_dir} 目录")


if __name__ == "__main__":
    # 模型路径
    model_path = "/root/autodl-tmp/Llama-2-7b-hf/"

    # 运行比较
    results = compare_quantization_methods(model_path)

    # 打印总结
    print("\n量化方法困惑度比较总结:")
    for result in results:
        print(f"{result['name']}: {result['perplexity']:.4f}")