import torch
import numpy as np
from scipy.stats import gaussian_kde





# 设置量化参数
n1 = 4  # n1 = 4 => b1 = 2^4 = 16
n2 = 3  # n2 = 3 => b2 = 2^3 = 8
b1 = 2 ** n1  # 第一次量化到 16 个值
b2 = 2 ** n2  # 第二次量化到 8 个值


half_range_b1 = b1 / 2

# [-((b2  / 2), ((b2  / 2)] 区间，该参数用于最终量化结果的映射
half_range_b2 = b2 / 2

def half_range(num_levels):
    return num_levels / 2


# 生成非均匀多峰分布的随机数
# def generate_data():
#     np.random.seed(42)
#     data = np.concatenate([
#         np.random.normal(loc=-31, scale=1.5, size=100),
#         np.random.normal(loc=-11, scale=1.5, size=1200),  # 第一个峰
#         np.random.normal(loc=-3, scale=0.8, size=2500),  # 第二个峰
#         np.random.normal(loc=0, scale=0.5, size=1000),   # 第三个峰
#         np.random.normal(loc=2, scale=0.3, size=2000),   # 第四个峰
#         np.random.normal(loc=4, scale=1.0, size=1500),   # 第五个峰
#         np.random.normal(loc=9, scale=0.9, size=1000),   # 第六个峰
#         np.random.normal(loc=15, scale=0.6, size=500),    # 第七个峰
#         np.random.normal(loc=27, scale=0.8, size=500),  # 第八个峰
#         np.random.normal(loc=50, scale=0.8, size=500)  # 第九个峰
#     ])
#     return torch.tensor(data)


def generate_data():
    np.random.seed(42)
    data = np.concatenate([
        np.random.normal(loc=-5, scale=1.0, size=3000),
        np.random.normal(loc=1, scale=0.5, size=4000),
        np.random.normal(loc=3, scale=0.8, size=3000)
    ])
    return torch.tensor(data)


# 量化函数
def quantize(data, num_levels):
    min_val = torch.min(data)
    max_val = torch.max(data)
    quantized_data = torch.round((data - min_val) / (max_val - min_val) * (num_levels - 1))
    return quantized_data


# 映射到 [-((b2 - 1) / 2), ((b2 - 1) / 2)] 区间的函数
# 映射错了，应该是[-b2 / 2), b2 / 2 - 1]
def map_to_target_range(data, original_min, original_max, target_min, target_max):
    # # 线性映射 错了
    # mapped_data = (data - original_min) / (original_max - original_min) * (target_max - target_min) + target_min

    # 线性映射
    mapped_data = (data / original_max) * (target_max - target_min) + target_min

    # # 反正是第2步量化时才做的，直接平移即可
    # offset = num_levels / 2
    # mapped_data = data - offset

    return mapped_data


#-------------------------反量化------------------------------#
def interval_based_dequantize(quantized_data, num_levels, min_val, max_val):
    """
    将量化后的数据反向映射回原始数据范围。
    
    参数:
        quantized_data: torch.Tensor，量化后的数据。
        num_levels: int，量化级别的数量。
        min_val: float，原始数据的最小值。
        max_val: float，原始数据的最大值。
        
    返回:
        dequantized_data: torch.Tensor，反量化后的数据。
    """
    # 计算区间的步长
    interval_size = (max_val - min_val) / num_levels
    
    # 反量化公式：quantized_data * interval_size + min_val + interval_size / 2
    dequantized_data = quantized_data * interval_size + min_val + interval_size / 2
    
    return dequantized_data
#-----------------------------------------------------------------#





#-----------------------------------------------------------------#
# # 基于均匀区间的量化函数
def interval_based_quantize(data, num_levels):
    min_val = torch.min(data)
    max_val = torch.max(data)
    # 计算区间的步长（范围分成 num_levels 块，如-800,800分成16块，步长100）
    interval_size = (max_val - min_val) / num_levels # 6-(-8)=14,14/16=0.875

    # # 将数据按照每个区间大小来进行量化
    # quantized_data = torch.floor((data - min_val ) / interval_size) # + interval_size/2
    # # 由于量化范围是 0 到 num_levels-1，所以需要限制边界
    # quantized_data = torch.clamp(quantized_data, 0, num_levels - 1)

    # 将数据按照每个区间大小来进行量化，并使用四舍五入
    quantized_data = torch.round((data - min_val - interval_size / 2) / interval_size)
    # 如：0-800，区间长度为100，那么(60-50)/100=0.1，四舍五入为0，(710-50)/100=6.6，四舍五入为7
    # 如：-400-400，区间长度为100，那么-400+400-50 / 100=-0.5四舍五入0
    # -150-(-400)-50=200, 200/100=2
    # 如-5，3-0.4375=2.5625, 2.5625/0.875=2.9=3
    # 如2, 10-0.4375=9.5625, 9.5625/0.875=10.9=11
    # 如3,11-0.4375=10.5625，10.5625/0.875=12.07=12
    #*如1,1-(-8)-0.4375=8.5625, 8.5625/0.875=9.786=10
    
    # 量化范围是[0,num_levels-1]闭区间
    quantized_data = torch.clamp(quantized_data, 0, num_levels - 1)

    # 将当前的量化数据 Q 从[0,b1-1]映射到 [-((b1 - 1) / 2), ((b1 - 1) / 2)] 区间
    #Q_mapped = map_to_target_range(quantized_data, 0, num_levels - 1, -half_range(num_levels), half_range(num_levels) - 1)
    #*如10，mapped_data = (10 / 15) * (7 - -8) + -8 = 2
    #*相当于原先的1通过区间的量化变成了2
    

    return quantized_data
#-----------------------------------------------------------------#




#------------------基于均匀区间的量化函数（补充反量化）---------------------#
# 基于均匀区间的量化函数（含反量化）
def interval_based_quantize_dequantize(data, num_levels, dequantize=True):
    """
    基于均匀区间的量化函数，支持反量化。

    参数:
        data: torch.Tensor，输入的浮点数据。
        num_levels: int，量化级别数量。
        dequantize: bool，是否进行反量化，默认为 False。

    返回:
        quantized_data: torch.Tensor，量化后的数据。
        dequantized_data: torch.Tensor，反量化后的数据（如果 dequantize=True）。
    """
    min_val = torch.min(data)
    max_val = torch.max(data)
    
    # 计算区间的步长（范围分成 num_levels 块）
    interval_size = (max_val - min_val) / num_levels  # 例如 14 / 16 = 0.875

    # 将数据按照每个区间大小来进行量化，并使用四舍五入
    quantized_data = torch.round((data - min_val - interval_size / 2) / interval_size)

    # 限制量化范围为 [0, num_levels - 1]
    quantized_data = torch.clamp(quantized_data, 0, num_levels - 1)

    # 反量化：将量化值映射回原始范围，再乘回去，加回去
    # 计算反量化的 scale 和 bias
    scale = interval_size
    bias = min_val + interval_size / 2
    dequantized_data = quantized_data * scale + bias

    return quantized_data, dequantized_data


def interval_based_quantize_dequantize_2(data, num_levels, dequantize=True):
    """
    基于均匀区间的量化函数，支持反量化。

    参数:
        data: torch.Tensor，输入的浮点数据。
        num_levels: int，量化级别数量。
        dequantize: bool，是否进行反量化，默认为 False。

    返回:
        quantized_data: torch.Tensor，量化后的数据。
        dequantized_data: torch.Tensor，反量化后的数据（如果 dequantize=True）。
    """
    min_val = torch.min(data)
    max_val = torch.max(data)

    # 计算区间的步长（范围分成 num_levels 块）
    interval_size = (max_val - min_val) / num_levels  # 例如 14 / 16 = 0.875

    # 将数据按照每个区间大小来进行量化，并使用四舍五入
    quantized_data = torch.round((data - min_val - interval_size / 2) / interval_size)

    # 限制量化范围为 [0, num_levels - 1]
    quantized_data = torch.clamp(quantized_data, 0, num_levels - 1)

    # 反量化：将量化值映射回原始范围，再乘回去，加回去
    # 计算反量化的 scale 和 bias
    scale = interval_size
    bias = min_val + interval_size / 2
    dequantized_data = quantized_data * scale + bias

    return quantized_data, dequantized_data, scale, bias
#-----------------------------------------------------------------#







#-----------------------------------------------------------------#
# （加映射版）基于均匀区间的量化函数,映射到[-8,7]区间
def interval_based_quantize_map(data, num_levels):
    min_val = torch.min(data)
    max_val = torch.max(data)
    # 计算区间的步长（范围分成 num_levels 块，如-800,800分成16块，步长100）
    interval_size = (max_val - min_val) / num_levels # 6-(-8)=14,14/16=0.875

    # 将数据按照每个区间大小来进行量化，并使用四舍五入
    quantized_data = torch.round((data - min_val - interval_size / 2) / interval_size)
    
    # 量化范围是[0,num_levels-1]闭区间
    quantized_data = torch.clamp(quantized_data, 0, num_levels - 1)

    # 将当前的量化数据 Q 从[0,b2-1]映射到 [-((b2 - 1) / 2), ((b2 - 1) / 2)] 区间
    Q_mapped = map_to_target_range(quantized_data, 0, num_levels - 1, -half_range(num_levels), half_range(num_levels) - 1)

    return Q_mapped
#-----------------------------------------------------------------#






#-----------------------------------------------------------------#
# （每一步打印版 不过其实没必要）基于均匀区间的量化函数,映射到[-8,7]区间
def interval_based_quantize_eachprint(data, num_levels):
    min_val = torch.min(data)
    max_val = torch.max(data)
    hist_data = []  # 用于保存每一步的直方图数据
    
    # 计算区间的步长（范围分成 num_levels-1 块）
    interval_size = (max_val - min_val) / num_levels


    # 将数据按照每个区间大小来进行量化，并使用四舍五入，如：0-800，区间长度为100，那么(60-50)/100=0.1，四舍五入为0，(710-50)/100=6.6，四舍五入为7
    quantized_data = torch.round((data - min_val - interval_size / 2) / interval_size)
    # 量化范围是[0,num_levels-1]闭区间
    quantized_data = torch.clamp(quantized_data, 0, num_levels - 1)

    # 将当前的量化数据 Q 从[0,b2-1]映射到 [-((b2 - 1) / 2), ((b2 - 1) / 2)] 区间
    Q_mapped = map_to_target_range(quantized_data, 0, num_levels - 1, -half_range(num_levels), half_range(num_levels) - 1)
    

    # map_to_target_range映射的范围不对，改一下，而且(b2 - 1) / 2要取个整

    # 保存映射后的直方图数据
    hist_data.append(Q_mapped.clone().numpy())

    return Q_mapped, hist_data
#-----------------------------------------------------------------#









#————————————————————————————关键量化函数————————————————————————————
# 动态量化级别缩减函数
# def dynamic_quantize_level_reduce(Q, D, num_levels):
#     unique_levels = torch.unique(Q)  # 提取 Q 中的唯一值
#     unique_levels.sort()  # 排序
#     iteration = 0  # 初始化循环计数器
#     hist_data = []  # 用于保存每一步的直方图数据
#
#     while len(unique_levels) > num_levels:
#         iteration += 1  # 循环次数加一
#         # 计算每一个level对应的误差
#         E = (Q - D) ** 2
#         # 计算每一个level对应的误差和
#         E_sum = {level.item(): sum(E[Q == level]).item() for level in unique_levels}
#         # 获得误差最小对应的level
#         min_level = min(E_sum, key=E_sum.get)
#         # 找到 min_level 在 unique_levels 中的索引位置
#         min_level_idx = torch.where(unique_levels == min_level)[0].item()
#
#         # 打印信息
#         print(f"Iteration {iteration}:")
#         print(f"unique_levels: {unique_levels}")
#         print(f"min_level_idx: {min_level_idx}")
#
#         # 设置默认值
#         less_level = None
#         more_level = None
#
#         if min_level_idx == 0:  # 如果最小误差对应的level是第一个
#             Q[Q == min_level] = unique_levels[1]
#         elif min_level_idx == len(unique_levels) - 1:# 如果最小误差对应的level是最后一个
#             Q[Q == min_level] = unique_levels[-2]
#         else:
#             less_level = unique_levels[min_level_idx - 1].item()
#             more_level = unique_levels[min_level_idx + 1].item()
#             Q[(Q == min_level) & (D <= min_level)] = less_level
#             Q[(Q == min_level) & (D > min_level)] = more_level
#
#         # 打印出被替换的level的值
#         print(f"less_level: {less_level}, more_level: {more_level}")
#
#         # 更新唯一值
#         unique_levels = torch.unique(Q)
#         unique_levels.sort()  # 排序
#
#         # 将当前的量化数据 Q 从[0, b1-1]映射到 [-b2 / 2, b2 / 2 - 1] 区间
#         Q_mapped = map_to_target_range(Q, 0, b1 - 1, -half_range(num_levels), half_range(num_levels) - 1)
#
#
#
#         # # 保存直方图数据
#         # hist_data.append(Q.clone().numpy())
#
#         # 保存映射后的直方图数据
#         hist_data.append(Q_mapped.clone().numpy())
#
#         # 打印出最小的误差，及其对应的level的值
#         print(f"min_level: {min_level}, min_error: {E_sum[min_level]}\n")
#
#     # # 全弄完之后，最后一次映射，将当前的量化数据 Q 从[0, b1-1]映射到 [-b2 / 2, b2 / 2 - 1] 区间
#     Q_mapped = map_to_target_range(Q, 0, b1 - 1, -half_range(num_levels), half_range(num_levels) - 1)
#
#
#     return Q_mapped, hist_data




# 按照学长建议修改，因为反量化以后的less_level和more_level与min_level的间隔应该不一样了
def dynamic_quantize_level_reduce(Q, D, num_levels):
    unique_levels = torch.unique(Q)  # 提取 Q 中的唯一值
    unique_levels.sort()  # 排序
    iteration = 0  # 初始化循环计数器
    hist_data = []  # 用于保存每一步的直方图数据

    while len(unique_levels) > num_levels:
        iteration += 1  # 循环次数加一
        # 计算每一个level对应的误差
        E = (Q - D) ** 2
        # 计算每一个level对应的误差和
        E_sum = {level.item(): sum(E[Q == level]).item() for level in unique_levels}
        # 获得误差最小对应的level
        min_level = min(E_sum, key=E_sum.get)
        # 找到unique_levels离minlevel最近的两个值
        min_level_idx = torch.where(unique_levels == min_level)[0].item()

        # 打印信息
        print(f"Iteration {iteration}:")
        print(f"unique_levels: {unique_levels}")
        print(f"min_level_idx: {min_level_idx}")

        # 设置默认值
        less_level = None
        more_level = None

        if min_level_idx == 0:  # 如果最小误差对应的level是第一个
            Q[Q == min_level] = unique_levels[1]
        elif min_level_idx == len(unique_levels) - 1:  # 如果最小误差对应的level是最后一个
            Q[Q == min_level] = unique_levels[-2]
        else:
            less_level = unique_levels[min_level_idx - 1].item()
            more_level = unique_levels[min_level_idx + 1].item()

            # 改进：按距离分配到 closer level
            Q[(Q == min_level) & (abs(D - less_level) < abs(D - more_level))] = less_level
            Q[(Q == min_level) & (abs(D - less_level) >= abs(D - more_level))] = more_level

        # 打印出被替换的level的值
        print(f"less_level: {less_level}, more_level: {more_level}")

        # 更新唯一值
        unique_levels = torch.unique(Q)
        unique_levels.sort()  # 排序

        # 保存直方图数据
        hist_data.append(Q.clone().numpy())

        # 打印出最小的误差，及其对应的level的值
        print(f"min_level: {min_level}, min_error: {E_sum[min_level]}\n")

    return Q, hist_data

#————————————————————————————————————————————————————————————————







#—————————————————动态量化级别缩减函数（加入反量化）————————————————
# 动态量化级别缩减函数（支持动态反量化）————测试效果目前最好

def dynamic_quantize_level_reduce_dequantize(Q, D, num_levels):
    unique_levels = torch.unique(Q)  # 提取 Q 中的唯一值
    unique_levels.sort()  # 排序
    iteration = 0  # 初始化循环计数器
    hist_data = []  # 用于保存每一步的直方图数据
    level_mapping = {}  # 动态映射字典，动态记录每个量化级别的原始值（用于反量化）

    # 初始化映射关系
    for level in unique_levels:
        level_mapping[level.item()] = level.item()

    while len(unique_levels) > num_levels:
        iteration += 1  # 循环次数加一
        # 计算每一个level对应的误差
        E = (Q - D) ** 2
        # 计算每一个level对应的误差和
        E_sum = {level.item(): sum(E[Q == level]).item() for level in unique_levels}

        # 打印每个 level 的误差
        print(f"Iteration {iteration}:")
        print("Loss for each level:")
        for idx, (level, loss) in enumerate(E_sum.items()):
            print(f"  [{idx}] Level {level}: {loss:.3f}")


        # 获得误差最小对应的level
        min_level = min(E_sum, key=E_sum.get)
        # 找到 min_level 在 unique_levels 中的索引位置
        min_level_idx = torch.where(unique_levels == min_level)[0].item()
        print(f"  -> Min loss level: {min_level}, Min loss level Index: [{min_level_idx:d}]")

        # 设置默认值
        less_level = None
        more_level = None

        if min_level_idx == 0:  # 如果最小误差对应的level是第一个
            replacement_level = unique_levels[1]
        elif min_level_idx == len(unique_levels) - 1:  # 如果最小误差对应的level是最后一个
            replacement_level = unique_levels[-2]
        else:
            less_level = unique_levels[min_level_idx - 1].item()
            more_level = unique_levels[min_level_idx + 1].item()

            #replacement_level = less_level if abs(min_level - less_level) < abs(min_level - more_level) else more_level
            # 基于反量化值的距离计算
            less_level_value = level_mapping[less_level]
            more_level_value = level_mapping[more_level]
            min_level_value = level_mapping[min_level]

            replacement_level = (
                less_level
                if abs(min_level_value - less_level_value) < abs(min_level_value - more_level_value)
                else more_level
            )


        # 更新 Q 中的值
        Q[Q == min_level] = replacement_level

        # 更新映射关系
        if isinstance(replacement_level, torch.Tensor):  # 检查类型
            replacement_level = replacement_level.item()  # 转换为 Python 数值

        # 更新 level_mapping
        level_mapping[replacement_level] = (level_mapping[replacement_level] + level_mapping[min_level]) / 2
        del level_mapping[min_level]  # 删除已合并的级别


        # 更新唯一值
        unique_levels = torch.unique(Q)
        unique_levels.sort()

        # 保存直方图数据
        hist_data.append(Q.clone().numpy())

        # 打印当前迭代信息
        print(f"Iteration {iteration}:")
        print(f"Updated level_mapping: {level_mapping}")

    # 使用最终的映射关系进行反量化
    dequantized_data = Q.clone().float()
    for quantized_level, original_value in level_mapping.items():
        dequantized_data[Q == quantized_level] = original_value

    return dequantized_data, hist_data


# def dynamic_quantize_level_reduce_dequantize(Q, D, num_levels):
#     unique_levels = torch.unique(Q)  # 提取 Q 中的唯一值
#     unique_levels.sort()  # 排序
#     iteration = 0  # 初始化循环计数器
#     hist_data = []  # 用于保存每一步的直方图数据
#     level_mapping = {}  # 动态映射字典，动态记录每个量化级别的原始值（用于反量化）
#
#     # 初始化映射关系
#     for level in unique_levels:
#         level_mapping[level.item()] = level.item()
#
#     while len(unique_levels) > num_levels:
#         iteration += 1  # 循环次数加一
#         # 计算每一个level对应的误差
#         E = (Q - D) ** 2
#         # 计算每一个level对应的误差和
#         E_sum = {level.item(): sum(E[Q == level]).item() for level in unique_levels}
#
#         # 找到误差最小的级别集合
#         min_error = min(E_sum.values())
#         min_levels = [level for level, error in E_sum.items() if error == min_error]
#
#         # 如果有多个候选的 min_level，选取离中心最远的
#         if len(min_levels) > 1:
#             center = (unique_levels.max() + unique_levels.min()) / 2  # 当前中心值
#             min_level = max(min_levels, key=lambda x: abs(x - center))  # 选离中心最远的
#         else:
#             min_level = min_levels[0]
#
#         # 找到 min_level 在 unique_levels 中的索引位置
#         min_level_idx = torch.where(unique_levels == min_level)[0].item()
#
#         # 设置默认值
#         less_level = None
#         more_level = None
#
#         if min_level_idx == 0:  # 如果最小误差对应的level是第一个
#             replacement_level = unique_levels[1]
#         elif min_level_idx == len(unique_levels) - 1:  # 如果最小误差对应的level是最后一个
#             replacement_level = unique_levels[-2]
#         else:
#             less_level = unique_levels[min_level_idx - 1].item()
#             more_level = unique_levels[min_level_idx + 1].item()
#
#             # 基于反量化值的距离计算
#             less_level_value = level_mapping[less_level]
#             more_level_value = level_mapping[more_level]
#             min_level_value = level_mapping[min_level]
#
#             replacement_level = (
#                 less_level
#                 if abs(min_level_value - less_level_value) < abs(min_level_value - more_level_value)
#                 else more_level
#             )
#
#         # 更新 Q 中的值
#         Q[Q == min_level] = replacement_level
#
#         # 更新映射关系
#         if isinstance(replacement_level, torch.Tensor):  # 检查类型
#             replacement_level = replacement_level.item()  # 转换为 Python 数值
#
#         # 更新 level_mapping
#         level_mapping[replacement_level] = (level_mapping[replacement_level] + level_mapping[min_level]) / 2
#         del level_mapping[min_level]  # 删除已合并的级别
#
#         # 更新唯一值
#         unique_levels = torch.unique(Q)
#         unique_levels.sort()
#
#         # 保存直方图数据
#         hist_data.append(Q.clone().numpy())
#
#         print(f"Iteration {iteration}:")
#         print(f"Updated level_mapping: {level_mapping}")
#
#     # 使用最终的映射关系进行反量化
#     dequantized_data = Q.clone().float()
#     for quantized_level, original_value in level_mapping.items():
#         dequantized_data[Q == quantized_level] = original_value
#
#     return dequantized_data, hist_data


#————————————————————————————————————————————————————————————————



#————————————————————————————————————————————————————————————————
#  动态量化级别缩减函数，支持动态反量化（调整后）
# def dynamic_quantize_level_reduce_dequantize(Q, D, num_levels):
#     unique_levels = torch.unique(Q)  # 提取 Q 中的唯一值
#     unique_levels.sort()  # 排序
#     iteration = 0  # 初始化循环计数器
#     hist_data = []  # 用于保存每一步的直方图数据
#     level_mapping = {}  # 动态记录每个量化级别的原始值（用于反量化）
#
#     # 初始化映射关系
#     for level in unique_levels:
#         level_mapping[level.item()] = [level.item()]  # 改为列表记录历史信息
#
#     while len(unique_levels) > num_levels:
#         iteration += 1  # 循环次数加一
#
#         # 计算每一个level对应的误差
#         E = (Q - D) ** 2
#         # 计算每一个level对应的误差和
#         E_sum = {level.item(): sum(E[Q == level]).item() for level in unique_levels}
#         # 获得误差最小对应的level
#         min_level = min(E_sum, key=E_sum.get)
#         # 找到 min_level 在 unique_levels 中的索引位置
#         min_level_idx = torch.where(unique_levels == min_level)[0].item()
#
#         # 设置默认值
#         less_level = None
#         more_level = None
#
#         # 根据 min_level_idx 位置，更新 Q 的值
#         if min_level_idx == 0:  # 如果最小误差对应的level是第一个
#             Q[Q == min_level] = unique_levels[1]
#             replacement_level = unique_levels[1].item()
#         elif min_level_idx == len(unique_levels) - 1:  # 如果最小误差对应的level是最后一个
#             Q[Q == min_level] = unique_levels[-2]
#             replacement_level = unique_levels[-2].item()
#         else:
#             # 找到相邻的量化级别
#             less_level = unique_levels[min_level_idx - 1].item()
#             more_level = unique_levels[min_level_idx + 1].item()
#
#             # 根据数据 D 的值决定替换到哪个级别
#             Q[(Q == min_level) & (D <= min_level)] = less_level
#             Q[(Q == min_level) & (D > min_level)] = more_level
#
#             # 动态确定替换的级别
#             replacement_level = less_level if (Q[(Q == min_level) & (D <= min_level)] == less_level).sum() > \
#                                               (Q[(Q == min_level) & (
#                                                           D > min_level)] == more_level).sum() else more_level
#
#         # 更新映射关系
#         if isinstance(replacement_level, torch.Tensor):  # 检查类型
#             replacement_level = replacement_level.item()  # 转换为 Python 数值
#
#         # 更新 level_mapping（合并替换级别的历史信息）
#         level_mapping[replacement_level].extend(level_mapping[min_level])
#         del level_mapping[min_level]  # 删除已合并的级别
#
#         # 更新唯一值
#         unique_levels = torch.unique(Q)
#         unique_levels.sort()
#
#         # 保存直方图数据
#         hist_data.append(Q.clone().numpy())
#
#         print(f"Iteration {iteration}:")
#         print(f"Removed level: {min_level}, Updated level_mapping: {level_mapping}")
#
#     # 使用最终的映射关系进行反量化
#     dequantized_data = Q.clone().float()
#     for quantized_level, original_values in level_mapping.items():
#         # 使用历史信息的均值作为反量化结果
#         dequantized_data[Q == quantized_level] = sum(original_values) / len(original_values)
#
#     return dequantized_data, hist_data


#————————————————————————————————————————————————————————————————


# 2024/11/21
# 反量化以后的less_level和more_level与min_level的间隔应该不一样，需要比较一下d-less_level和(less_level+more_level)/2也就是和中点比
# Q[(Q == min_level) & (D - less_level_value < more_level_value - D)] = less_level
# 算了干脆一步到位，更精确一点，不用中点间接看了，直接按照距离来分配不就行了

# def dynamic_quantize_level_reduce_dequantize(Q, D, num_levels):
#     """
#     动态量化级别缩减函数，支持动态反量化（基于距离分配的改进版）。
#
#     参数:
#         Q: torch.Tensor, 初始量化后的数据。
#         D: torch.Tensor, 原始数据，用于计算误差。
#         num_levels: int, 最终保留的量化级别数量。
#
#     返回:
#         dequantized_data: torch.Tensor, 动态反量化后的数据。
#         hist_data: list, 每次迭代后 Q 的分布。
#     """
#     unique_levels = torch.unique(Q)  # 提取 Q 中的唯一值
#     unique_levels.sort()  # 排序
#     iteration = 0  # 初始化循环计数器
#     hist_data = []  # 用于保存每一步的直方图数据
#     level_mapping = {}  # 动态记录每个量化级别的原始值（用于反量化）
#
#
#     # 初始化映射关系
#     for level in unique_levels:
#         level_mapping[level.item()] = [level.item()]  # 改为列表记录历史信息
#
#     while len(unique_levels) > num_levels:
#         iteration += 1  # 循环次数加一
#
#         # 计算每一个level对应的误差
#         E = (Q - D) ** 2
#         # 计算每一个level对应的误差和
#         E_sum = {level.item(): sum(E[Q == level]).item() for level in unique_levels}
#         # 获得误差最小对应的level
#         min_level = min(E_sum, key=E_sum.get)
#         # 找到 min_level 在 unique_levels 中的索引位置
#         min_level_idx = torch.where(unique_levels == min_level)[0].item()
#
#         # 设置默认值
#         less_level = None
#         more_level = None
#
#         # 根据 min_level_idx 位置，更新 Q 的值
#         if min_level_idx == 0:  # 如果最小误差对应的level是第一个
#             Q[Q == min_level] = unique_levels[1]
#             replacement_level = unique_levels[1].item()
#         elif min_level_idx == len(unique_levels) - 1:  # 如果最小误差对应的level是最后一个
#             Q[Q == min_level] = unique_levels[-2]
#             replacement_level = unique_levels[-2].item()
#         else:
#             # 找到相邻的量化级别
#             less_level = unique_levels[min_level_idx - 1].item()
#             more_level = unique_levels[min_level_idx + 1].item()
#
#             # 获取反量化后的值
#
#
#             # 方法1：均值
#             less_level_value = sum(level_mapping[less_level]) / len(level_mapping[less_level])
#             more_level_value = sum(level_mapping[more_level]) / len(level_mapping[more_level])
#
#
#             # 根据距离动态分配到更接近的级别
#             Q[(Q == min_level) & (abs(D - less_level_value) < abs(D - more_level_value))] = less_level
#             Q[(Q == min_level) & (abs(D - less_level_value) >= abs(D - more_level_value))] = more_level
#
#             # 动态确定替换的级别
#             replacement_level = less_level if abs(min_level - less_level) < abs(min_level - more_level) else more_level
#
#         # 更新映射关系
#         if isinstance(replacement_level, torch.Tensor):  # 检查类型
#             replacement_level = replacement_level.item()  # 转换为 Python 数值
#
#         # 合并映射关系，历史均值动态调整
#         level_mapping[replacement_level].extend(level_mapping[min_level])
#         del level_mapping[min_level]  # 删除已合并的级别
#
#         # 更新唯一值
#         unique_levels = torch.unique(Q)
#         unique_levels.sort()
#
#         # 保存直方图数据
#         hist_data.append(Q.clone().numpy())
#
#         print(f"Iteration {iteration}:")
#         print(f"Removed level: {min_level}, Updated level_mapping: {level_mapping}")
#
#     # 使用最终的映射关系进行反量化
#     dequantized_data = Q.clone().float()
#     for quantized_level, original_values in level_mapping.items():
#         # 使用历史信息的均值作为反量化结果
#         dequantized_data[Q == quantized_level] = sum(original_values) / len(original_values)
#
#     return dequantized_data, hist_data


# def dynamic_quantize_level_reduce_dequantize(Q, D, num_levels):
#     """
#     动态量化级别缩减函数，支持动态反量化（改进版，无固定 replacement_level）。
#
#     参数:
#         Q: torch.Tensor, 初始量化后的数据。
#         D: torch.Tensor, 原始数据，用于计算误差。
#         num_levels: int, 最终保留的量化级别数量。
#
#     返回:
#         dequantized_data: torch.Tensor, 动态反量化后的数据。
#         hist_data: list, 每次迭代后 Q 的分布。
#     """
#     unique_levels = torch.unique(Q)  # 提取 Q 中的唯一值
#     unique_levels.sort()  # 排序
#     iteration = 0  # 初始化循环计数器
#     hist_data = []  # 用于保存每一步的直方图数据
#     level_mapping = {}  # 动态映射字典，动态记录每个量化级别的原始值（用于反量化）
#
#     # 初始化映射关系
#     for level in unique_levels:
#         level_mapping[level.item()] = [level.item()]  # 改为列表记录历史信息
#
#     while len(unique_levels) > num_levels:
#         iteration += 1  # 循环次数加一
#
#         # 计算每一个level对应的误差
#         E = (Q - D) ** 2
#         # 计算每一个level对应的误差和
#         E_sum = {level.item(): sum(E[Q == level]).item() for level in unique_levels}
#         # 获得误差最小对应的level
#         min_level = min(E_sum, key=E_sum.get)
#         # 找到 min_level 在 unique_levels 中的索引位置
#         min_level_idx = torch.where(unique_levels == min_level)[0].item()
#
#         # 设置默认值
#         less_level = None
#         more_level = None
#
#         # 根据 min_level_idx 位置，更新 Q 的值
#         if min_level_idx == 0:  # 如果最小误差对应的level是第一个
#             Q[Q == min_level] = unique_levels[1]
#             less_level = unique_levels[1].item()
#         elif min_level_idx == len(unique_levels) - 1:  # 如果最小误差对应的level是最后一个
#             Q[Q == min_level] = unique_levels[-2]
#             more_level = unique_levels[-2].item()
#         else:
#             # 找到相邻的量化级别
#             less_level = unique_levels[min_level_idx - 1].item()
#             more_level = unique_levels[min_level_idx + 1].item()
#
#             # 获取反量化后的值
#             less_level_value = sum(level_mapping[less_level]) / len(level_mapping[less_level])
#             more_level_value = sum(level_mapping[more_level]) / len(level_mapping[more_level])
#
#             # 根据距离动态分配到更接近的级别
#             Q[(Q == min_level) & (abs(D - less_level_value) < abs(D - more_level_value))] = less_level
#             Q[(Q == min_level) & (abs(D - less_level_value) >= abs(D - more_level_value))] = more_level
#
#         # 动态更新 level_mapping
#         if less_level is not None:
#             level_mapping[less_level].extend(level_mapping[min_level])
#         if more_level is not None:
#             level_mapping[more_level].extend(level_mapping[min_level])
#         del level_mapping[min_level]  # 删除已合并的级别
#
#         # 更新唯一值
#         unique_levels = torch.unique(Q)
#         unique_levels.sort()
#
#         # 保存直方图数据
#         hist_data.append(Q.clone().numpy())
#
#         print(f"Iteration {iteration}:")
#         print(f"Removed level: {min_level}, Updated level_mapping: {level_mapping}")
#
#     # 使用最终的映射关系进行反量化
#     dequantized_data = Q.clone().float()
#     for quantized_level, original_values in level_mapping.items():
#         # 使用历史信息的均值作为反量化结果，历史均值可以有效地平滑分布，避免偏向最后一次合并的值
#         dequantized_data[Q == quantized_level] = sum(original_values) / len(original_values)
#
#     return dequantized_data, hist_data



#————————————————————————————————————————————————————————————————




#————————————————————————————————————————————————————————————————
# 去掉本level后算其他误差的代码版本
# def dynamic_quantize_level_reduce_dequantize_2(Q, D, num_levels):
#     """
#     动态量化级别缩减函数（第二版本，支持动态反量化）。
#
#     参数:
#         Q: torch.Tensor, 初始量化后的数据。
#         D: torch.Tensor, 原始数据，用于计算误差。
#         num_levels: int, 最终保留的量化级别数量。
#
#     返回:
#         dequantized_data: torch.Tensor, 动态反量化后的数据。
#         hist_data: list, 每次迭代后 Q 的分布。
#     """
#     unique_levels = torch.unique(Q)  # 提取 Q 中的唯一值
#     unique_levels.sort()  # 排序
#     iteration = 0  # 初始化循环计数器
#     hist_data = []  # 用于保存每一步的直方图数据
#     level_mapping = {}  # 动态映射字典，动态记录每个量化级别的原始值（用于反量化）
#
#     # 初始化映射关系
#     for level in unique_levels:
#         level_mapping[level.item()] = level.item()
#
#     while len(unique_levels) > num_levels:
#         iteration += 1  # 循环次数加一
#
#         total_errors = {}  # 用于存储去掉每个level后的总误差
#
#         # 遍历每个level，模拟其被移除后的情况
#         for min_level in unique_levels:
#             Q_copy = Q.clone()  # 创建 Q 的副本
#
#             # 找到 min_level 的索引位置
#             min_level_idx = torch.where(unique_levels == min_level)[0].item()
#
#             # 设置默认值
#             less_level = None
#             more_level = None
#
#             # 模拟移除逻辑
#             if min_level_idx == 0:  # 如果是第一个级别
#                 Q_copy[Q == min_level] = unique_levels[1]
#             elif min_level_idx == len(unique_levels) - 1:  # 如果是最后一个级别
#                 Q_copy[Q == min_level] = unique_levels[-2]
#             else:  # 如果是中间级别
#                 less_level = unique_levels[min_level_idx - 1].item()
#                 more_level = unique_levels[min_level_idx + 1].item()
#
#                 # 根据数据 D 的值决定替换到哪个级别
#                 Q_copy[(Q == min_level) & (D <= min_level)] = less_level
#                 Q_copy[(Q == min_level) & (D > min_level)] = more_level
#
#             # 计算移除当前level后的总误差
#             total_error = ((Q_copy - D) ** 2).sum().item()
#             total_errors[min_level.item()] = total_error
#
#         # 找到移除后误差最小的量化级别
#         level_to_remove = min(total_errors, key=total_errors.get)
#
#         # 找到被移除级别的索引位置
#         level_to_remove_idx = torch.where(unique_levels == level_to_remove)[0].item()
#
#         # 确定替换后的级别
#         if level_to_remove_idx == 0:  # 如果是第一个级别
#             Q[Q == level_to_remove] = unique_levels[1]
#             replacement_level = unique_levels[1].item()
#         elif level_to_remove_idx == len(unique_levels) - 1:  # 如果是最后一个级别
#             Q[Q == level_to_remove] = unique_levels[-2]
#             replacement_level = unique_levels[-2].item()
#         else:  # 如果是中间级别
#             less_level = unique_levels[level_to_remove_idx - 1].item()
#             more_level = unique_levels[level_to_remove_idx + 1].item()
#
#             # 根据数据 D 的值决定替换到哪个级别
#             Q[(Q == level_to_remove) & (D <= level_to_remove)] = less_level
#             Q[(Q == level_to_remove) & (D > level_to_remove)] = more_level
#
#             # 动态确定替换的级别
#             replacement_level = less_level if (Q[(Q == level_to_remove) & (D <= level_to_remove)] == less_level).sum() > \
#                                              (Q[(Q == level_to_remove) & (D > level_to_remove)] == more_level).sum() else more_level
#
#         # 更新映射关系
#         level_mapping[replacement_level] = (level_mapping[replacement_level] + level_mapping[level_to_remove]) / 2
#         del level_mapping[level_to_remove]  # 删除已合并的级别
#
#         # 更新唯一值
#         unique_levels = torch.unique(Q)
#         unique_levels.sort()
#
#         # 保存直方图数据
#         hist_data.append(Q.clone().numpy())
#
#         print(f"Iteration {iteration}:")
#         print(f"Removed level: {level_to_remove}, Updated level_mapping: {level_mapping}")
#
#     # 使用最终的映射关系进行反量化
#     dequantized_data = Q.clone().float()
#     for quantized_level, original_value in level_mapping.items():
#         dequantized_data[Q == quantized_level] = original_value
#
#     return dequantized_data, hist_data

#————————————————————————————————————————————————————————————————




#————————————————————————————————————————————————————————————————
# def dynamic_quantize_level_reduce_dequantize_2(Q, D, num_levels):
#     """
#     动态量化级别缩减函数（基于距离分配的改进版，支持动态反量化）。
#
#     参数:
#         Q: torch.Tensor, 初始量化后的数据。
#         D: torch.Tensor, 原始数据，用于计算误差。
#         num_levels: int, 最终保留的量化级别数量。
#
#     返回:
#         dequantized_data: torch.Tensor, 动态反量化后的数据。
#         hist_data: list, 每次迭代后 Q 的分布。
#     """
#     unique_levels = torch.unique(Q)  # 提取 Q 中的唯一值
#     unique_levels.sort()  # 排序
#     iteration = 0  # 初始化循环计数器
#     hist_data = []  # 用于保存每一步的直方图数据
#     level_mapping = {}  # 动态映射字典，动态记录每个量化级别的原始值（用于反量化）
#
#     # 初始化映射关系
#     for level in unique_levels:
#         level_mapping[level.item()] = [level.item()]  # 改为列表记录历史信息
#
#     while len(unique_levels) > num_levels:
#         iteration += 1  # 循环次数加一
#
#         total_errors = {}  # 用于存储去掉每个level后的总误差
#
#         # 遍历每个level，模拟其被移除后的情况
#         for min_level in unique_levels:
#             Q_copy = Q.clone()  # 创建 Q 的副本
#
#             # 找到 min_level 的索引位置
#             min_level_idx = torch.where(unique_levels == min_level)[0].item()
#
#             # 设置默认值
#             less_level = None
#             more_level = None
#
#             # 模拟移除逻辑
#             if min_level_idx == 0:  # 如果是第一个级别
#                 Q_copy[Q == min_level] = unique_levels[1]
#             elif min_level_idx == len(unique_levels) - 1:  # 如果是最后一个级别
#                 Q_copy[Q == min_level] = unique_levels[-2]
#             else:  # 如果是中间级别
#                 less_level = unique_levels[min_level_idx - 1].item()
#                 more_level = unique_levels[min_level_idx + 1].item()
#
#                 # 获取反量化后的值
#                 less_level_value = sum(level_mapping[less_level]) / len(level_mapping[less_level])
#                 more_level_value = sum(level_mapping[more_level]) / len(level_mapping[more_level])
#
#                 # 根据距离动态分配到更接近的级别
#                 Q_copy[(Q == min_level) & (abs(D - less_level_value) < abs(D - more_level_value))] = less_level
#                 Q_copy[(Q == min_level) & (abs(D - less_level_value) >= abs(D - more_level_value))] = more_level
#
#             # 计算移除当前level后的总误差
#             total_error = ((Q_copy - D) ** 2).sum().item()
#             total_errors[min_level.item()] = total_error
#
#         # 找到移除后误差最小的量化级别
#         level_to_remove = min(total_errors, key=total_errors.get)
#
#         # 找到被移除级别的索引位置
#         level_to_remove_idx = torch.where(unique_levels == level_to_remove)[0].item()
#
#         # 确定替换后的级别
#         if level_to_remove_idx == 0:  # 如果是第一个级别
#             Q[Q == level_to_remove] = unique_levels[1]
#             replacement_level = unique_levels[1].item()
#         elif level_to_remove_idx == len(unique_levels) - 1:  # 如果是最后一个级别
#             Q[Q == level_to_remove] = unique_levels[-2]
#             replacement_level = unique_levels[-2].item()
#         else:  # 如果是中间级别
#             less_level = unique_levels[level_to_remove_idx - 1].item()
#             more_level = unique_levels[level_to_remove_idx + 1].item()
#
#             # 获取反量化后的值
#             less_level_value = sum(level_mapping[less_level]) / len(level_mapping[less_level])
#             more_level_value = sum(level_mapping[more_level]) / len(level_mapping[more_level])
#
#             # 根据距离动态分配到更接近的级别
#             Q[(Q == level_to_remove) & (abs(D - less_level_value) < abs(D - more_level_value))] = less_level
#             Q[(Q == level_to_remove) & (abs(D - less_level_value) >= abs(D - more_level_value))] = more_level
#
#             # 动态确定替换的级别
#             replacement_level = less_level if (abs(D - less_level_value) < abs(D - more_level_value)).sum() > \
#                                               (abs(D - more_level_value) <= abs(
#                                                   D - less_level_value)).sum() else more_level
#
#         # 更新映射关系
#         level_mapping[replacement_level].extend(level_mapping[level_to_remove])
#         del level_mapping[level_to_remove]  # 删除已合并的级别
#
#         # 更新唯一值
#         unique_levels = torch.unique(Q)
#         unique_levels.sort()
#
#         # 保存直方图数据
#         hist_data.append(Q.clone().numpy())
#
#         print(f"Iteration {iteration}:")
#         print(f"Removed level: {level_to_remove}, Updated level_mapping: {level_mapping}")
#
#     # 使用最终的映射关系进行反量化
#     dequantized_data = Q.clone().float()
#     for quantized_level, original_values in level_mapping.items():
#         # 使用历史信息的均值作为反量化结果
#         dequantized_data[Q == quantized_level] = sum(original_values) / len(original_values)
#
#     return dequantized_data, hist_data

#————————————————————————————————————————————————————————————————


#————————————————————————————————————————————————————————————————
# 终极版，去掉本level后算其他误差的代码版本，n1=4，反量化距离优化版
# 目前效果最好的版本，只有多峰分布不够好，其他全都比均匀量化好
def dynamic_quantize_level_reduce_dequantize_2(Q, D, num_levels):
    unique_levels = torch.unique(Q)  # 提取 Q 中的唯一值
    unique_levels.sort()  # 排序
    iteration = 0  # 初始化循环计数器
    hist_data = []  # 用于保存每一步的直方图数据
    level_mapping = {}  # 动态映射字典，动态记录每个量化级别的原始值（用于反量化）

    # 初始化映射关系，将键转换为浮点数
    for level in unique_levels:
        level_mapping[float(level.item())] = float(level.item())

    while len(unique_levels) > num_levels:
        iteration += 1  # 循环次数加一

        total_errors = {}  # 用于存储去掉每个level后的总误差

        # 遍历每个level，模拟其被移除后的情况
        for min_level in unique_levels:
            min_level = float(min_level.item())  # 确保是浮点数
            Q_copy = Q.clone()  # 创建 Q 的副本
            # 找到 min_level 的索引位置
            min_level_idx = torch.where(unique_levels == min_level)[0].item()

            # 设置默认值
            less_level = None
            more_level = None

            # 模拟移除逻辑
            if min_level_idx == 0:  # 如果是第一个级别
                replacement_level = unique_levels[1]
            elif min_level_idx == len(unique_levels) - 1:  # 如果是最后一个级别
                replacement_level = unique_levels[-2]
            else:  # 如果是中间级别
                less_level = float(unique_levels[min_level_idx - 1].item())
                more_level = float(unique_levels[min_level_idx + 1].item())

                # 基于反量化值的距离计算
                less_level_value = level_mapping[less_level]
                more_level_value = level_mapping[more_level]
                min_level_value = level_mapping[min_level]

                replacement_level = (
                    less_level
                    if abs(min_level_value - less_level_value) < abs(min_level_value - more_level_value)
                    else more_level
                )

            # 更新 Q_copy 中的值
            Q_copy[Q == min_level] = replacement_level

            # 计算移除当前level后的总误差
            total_error = ((Q_copy - D) ** 2).sum().item()
            total_errors[min_level] = total_error

        # 找到移除后误差最小的量化级别
        level_to_remove = min(total_errors, key=total_errors.get)
        level_to_remove_loss = total_errors[level_to_remove]
        print(f"Iteration {iteration}:")
        print("Loss for each level after removing:")
        for idx, (level, loss) in enumerate(total_errors.items()):
            print(f"  [{idx}] Level {level}: {loss:.3f}")
        print(f"  -> Min loss level: {level_to_remove}, Loss: {level_to_remove_loss:.3f}")

        # 找到被移除级别的索引位置
        level_to_remove_idx = torch.where(unique_levels == level_to_remove)[0].item()

        # 确定替换后的级别
        if level_to_remove_idx == 0:  # 如果是第一个级别
            replacement_level = unique_levels[1]
        elif level_to_remove_idx == len(unique_levels) - 1:  # 如果是最后一个级别
            replacement_level = unique_levels[-2]
        else:  # 如果是中间级别
            less_level = float(unique_levels[level_to_remove_idx - 1].item())
            more_level = float(unique_levels[level_to_remove_idx + 1].item())

            # 基于反量化值的距离计算
            less_level_value = level_mapping[less_level]
            more_level_value = level_mapping[more_level]
            level_to_remove_value = level_mapping[level_to_remove]

            replacement_level = (
                less_level
                if abs(level_to_remove_value - less_level_value) < abs(level_to_remove_value - more_level_value)
                else more_level
            )

        # 更新 Q 中的值
        Q[Q == level_to_remove] = replacement_level

        # 更新映射关系
        replacement_level = float(replacement_level)  # 确保键为浮点数
        level_mapping[replacement_level] = (level_mapping[replacement_level] + level_mapping[level_to_remove]) / 2
        del level_mapping[level_to_remove]  # 删除已合并的级别

        # 更新唯一值
        unique_levels = torch.unique(Q)
        unique_levels.sort()

        # 保存直方图数据
        hist_data.append(Q.clone().numpy())

        # 打印当前迭代信息
        print(f"  Removed level {level_to_remove}, replaced with {replacement_level}")
        print(f"  Updated level_mapping: {level_mapping}\n")

    # 使用最终的映射关系进行反量化
    dequantized_data = Q.clone().float()
    for quantized_level, original_value in level_mapping.items():
        quantized_level = float(quantized_level)  # 确保类型一致
        dequantized_data[Q == quantized_level] = original_value

    return dequantized_data, hist_data
#————————————————————————————————————————————————————————————————
