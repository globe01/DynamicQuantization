import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from typing import List, Dict, Optional, Tuple
import time
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
from torch.utils.hooks import RemovableHandle

from real_quantization_utils import (
    interval_based_quantize_dequantize,
    new_dynamic_quantize,
    plot_quantization_results,
    save_quantization_results
)


class ActivationExtractor:
    def __init__(self, model_path: str):
        """
        初始化激活值提取器

        Args:
            model_path: LLaMA模型路径
        """
        self.model = LlamaForCausalLM.from_pretrained(model_path)
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
        self.handles = []
        self.activations = {}

    def _hook_fn(self, name: str):
        """创建钩子函数来捕获激活值"""

        def hook(module, input, output):
            self.activations[name] = output.detach()

        return hook

    def attach_hooks(self, layer_num: int):
        """
        为指定层添加钩子

        Args:
            layer_num: 要提取激活值的层号
        """
        layer = self.model.model.layers[layer_num]

        # 注册attention激活值的钩子
        self.handles.append(layer.self_attn.q_proj.register_forward_hook(
            self._hook_fn(f'layer_{layer_num}_q_activation')))
        self.handles.append(layer.self_attn.k_proj.register_forward_hook(
            self._hook_fn(f'layer_{layer_num}_k_activation')))
        self.handles.append(layer.self_attn.v_proj.register_forward_hook(
            self._hook_fn(f'layer_{layer_num}_v_activation')))

        # 注册FFN激活值的钩子
        self.handles.append(layer.mlp.gate_proj.register_forward_hook(
            self._hook_fn(f'layer_{layer_num}_ffn_gate_activation')))
        self.handles.append(layer.mlp.up_proj.register_forward_hook(
            self._hook_fn(f'layer_{layer_num}_ffn_up_activation')))
        self.handles.append(layer.mlp.down_proj.register_forward_hook(
            self._hook_fn(f'layer_{layer_num}_ffn_down_activation')))

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
        # 清空之前的激活值
        self.activations = {}

        # 添加钩子
        self.attach_hooks(layer_num)

        # tokenize输入文本
        inputs = self.tokenizer(input_text, return_tensors="pt")

        # 前向传播
        with torch.no_grad():
            self.model(**inputs)

        # 移除钩子
        self.remove_hooks()

        return self.activations


def quantize_and_evaluate(activation_tensor: torch.Tensor,
                          initial_levels: int,
                          target_levels: int,
                          name: str,
                          save_dir: str) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    """对激活值进行量化并评估结果"""
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 动态量化
    final_dequantized, hist_data, quantized_data, scale, bias = new_dynamic_quantize(
        data=activation_tensor,
        initial_levels=initial_levels,
        target_levels=target_levels,
        verbose=True
    )

    # 均匀量化对比
    uniform_quantized, uniform_dequantized = interval_based_quantize_dequantize(
        data=activation_tensor,
        num_levels=target_levels
    )

    # 计算误差
    dynamic_error = ((final_dequantized - activation_tensor) ** 2).mean().item()
    uniform_error = ((uniform_dequantized - activation_tensor) ** 2).mean().item()

    # 保存量化结果可视化
    plot_quantization_results(
        original=activation_tensor,
        first_quant=torch.tensor(hist_data[0]),
        second_quant=final_dequantized,
        layer_name=name,  # 参数名是 layer_name
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


def main():
    # 配置参数
    model_path = "C:/Users/globe/.llama/checkpoints/Llama-2-7b-hf"  # 用本地模型路径
    save_base_dir = "activation_quantization_results"
    input_text = "Hello, I am a student from Nankai University."  # 输入文本，用于触发模型的前向传播，生成激活值

    # 输入文本通过 tokenizer 进行编码，转化为输入张量，然后送入模型执行前向传播
    # 通过注册 钩子函数（hooks），提取指定层的激活值

    # 创建激活值提取器
    extractor = ActivationExtractor(model_path)

    total_start_time = time.time()

    # 为每一层进行量化
    for layer_num in range(32):# 32即为0-31层
        layer_start_time = time.time()
        print(f"\n{'=' * 50}")
        print(f"开始处理第 {layer_num} 层激活值 (总共32层)")
        print(f"{'=' * 50}")

        # 为每一层创建单独的保存目录
        save_dir = f"{save_base_dir}/layer_{layer_num:02d}"
        os.makedirs(save_dir, exist_ok=True)

        # 提取激活值
        activations = extractor.extract_activations(layer_num, input_text)

        # 对每个激活值进行量化
        for name, activation in activations.items():
            print(f"\n处理激活值: {name}")
            print(f"激活值形状: {activation.shape}")

            # 量化并评估
            dynamic_result, uniform_result, dynamic_error, uniform_error = quantize_and_evaluate(
                activation_tensor=activation,
                initial_levels=16,  # 2^4
                target_levels=8,  # 2^3
                name=name.replace(".", "_"),
                save_dir=save_dir
            )

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