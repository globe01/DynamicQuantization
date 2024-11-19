import torch
import numpy as np





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
    dequantized_data = quantized_data * interval_size + min_val + interval_size / 2

    return quantized_data, dequantized_data

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
        # 找到 min_level 在 unique_levels 中的索引位置
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
        elif min_level_idx == len(unique_levels) - 1:# 如果最小误差对应的level是最后一个
            Q[Q == min_level] = unique_levels[-2]
        else:
            less_level = unique_levels[min_level_idx - 1].item()
            more_level = unique_levels[min_level_idx + 1].item()
            Q[(Q == min_level) & (D <= min_level)] = less_level
            Q[(Q == min_level) & (D > min_level)] = more_level

        # 打印出被替换的level的值
        print(f"less_level: {less_level}, more_level: {more_level}")

        # 更新唯一值
        unique_levels = torch.unique(Q)
        unique_levels.sort()  # 排序

        # 将当前的量化数据 Q 从[0, b1-1]映射到 [-b2 / 2, b2 / 2 - 1] 区间
        Q_mapped = map_to_target_range(Q, 0, b1 - 1, -half_range(num_levels), half_range(num_levels) - 1)



        # # 保存直方图数据
        # hist_data.append(Q.clone().numpy())

        # 保存映射后的直方图数据
        hist_data.append(Q_mapped.clone().numpy())

        # 打印出最小的误差，及其对应的level的值
        print(f"min_level: {min_level}, min_error: {E_sum[min_level]}\n")

    # # 全弄完之后，最后一次映射，将当前的量化数据 Q 从[0, b1-1]映射到 [-b2 / 2, b2 / 2 - 1] 区间
    Q_mapped = map_to_target_range(Q, 0, b1 - 1, -half_range(num_levels), half_range(num_levels) - 1)


    return Q_mapped, hist_data

#————————————————————————————————————————————————————————————————





#—————————————————动态量化级别缩减函数（加入反量化）————————————————
# 动态量化级别缩减函数
# 动态量化级别缩减函数（支持动态反量化）
def dynamic_quantize_level_reduce_dequantize(Q, D, num_levels):
    unique_levels = torch.unique(Q)  # 提取 Q 中的唯一值
    unique_levels.sort()  # 排序
    iteration = 0  # 初始化循环计数器
    hist_data = []  # 用于保存每一步的直方图数据
    level_mapping = {}  # 动态记录每个量化级别的原始值（用于反量化）

    # 初始化映射关系
    for level in unique_levels:
        level_mapping[level.item()] = level.item()

    while len(unique_levels) > num_levels:
        iteration += 1  # 循环次数加一
        # 计算每一个level对应的误差
        E = (Q - D) ** 2
        # 计算每一个level对应的误差和
        E_sum = {level.item(): sum(E[Q == level]).item() for level in unique_levels}
        # 获得误差最小对应的level
        min_level = min(E_sum, key=E_sum.get)
        # 找到 min_level 在 unique_levels 中的索引位置
        min_level_idx = torch.where(unique_levels == min_level)[0].item()

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
            replacement_level = less_level if abs(min_level - less_level) < abs(min_level - more_level) else more_level

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

        print(f"Iteration {iteration}:")
        print(f"Updated level_mapping: {level_mapping}")

    # 使用最终的映射关系进行反量化
    dequantized_data = Q.clone().float()
    for quantized_level, original_value in level_mapping.items():
        dequantized_data[Q == quantized_level] = original_value

    return dequantized_data, hist_data

#————————————————————————————————————————————————————————————————










# def dynamic_quantize_level_reduce(Q, D, num_levels):
#     unique_levels = torch.unique(Q)  # 提取 Q 中的唯一值
#     unique_levels.sort()  # 排序
#     iteration = 0  # 初始化循环计数器
#     hist_data = []  # 用于保存每一步的直方图数据
#
#     while len(unique_levels) > num_levels:
#         iteration += 1  # 循环次数加一
#         total_errors = {}  # 用于保存去除每个 level 后的误差总和
#
#         # 遍历每个 level，计算去掉该 level 后的误差总和
#         for level in unique_levels:
#             # 创建一个副本，在副本中移除当前级别的元素
#             Q_copy = Q.clone()
#
#             # 获取当前级别在 unique_levels 中的索引位置
#             level_idx = torch.where(unique_levels == level)[0].item()
#
#             # 根据索引位置将当前级别的值重新分配给相邻的级别
#             if level_idx == 0:
#                 # 如果是第一个量化级别，重新分配到下一个级别
#                 Q_copy[Q == level] = unique_levels[1]
#             elif level_idx == len(unique_levels) - 1:
#                 # 如果是最后一个量化级别，重新分配到上一个级别
#                 Q_copy[Q == level] = unique_levels[-2]
#             else:
#                 # 否则，分配到相邻的两个级别，根据原始数据 D 的距离进行分配
#                 less_level = unique_levels[level_idx - 1].item()
#                 more_level = unique_levels[level_idx + 1].item()
#                 Q_copy[(Q == level) & (D <= level)] = less_level
#                 Q_copy[(Q == level) & (D > level)] = more_level
#
#             # 计算去除当前级别后的误差总和
#             E_copy = (Q_copy - D) ** 2
#             total_errors[level.item()] = E_copy.sum().item()
#
#         # 找到移除后误差总和最小的量化级别
#         level_to_remove = min(total_errors, key=total_errors.get)
#         print(
#             f"Iteration {iteration}: Removing level {level_to_remove} with total error {total_errors[level_to_remove]}")
#
#         # 在 Q 中移除误差最小的 level，并重新分配对应的元素
#         level_idx_to_remove = torch.where(unique_levels == level_to_remove)[0].item()
#
#         # 更新 Q 中的移除逻辑
#         if level_idx_to_remove == 0:
#             Q[Q == level_to_remove] = unique_levels[1]
#         elif level_idx_to_remove == len(unique_levels) - 1:
#             Q[Q == level_to_remove] = unique_levels[-2]
#         else:
#             less_level = unique_levels[level_idx_to_remove - 1].item()
#             more_level = unique_levels[level_idx_to_remove + 1].item()
#             Q[(Q == level_to_remove) & (D <= level_to_remove)] = less_level
#             Q[(Q == level_to_remove) & (D > level_to_remove)] = more_level
#
#         # 更新 unique_levels
#         unique_levels = torch.unique(Q)
#         unique_levels.sort()  # 排序
#
#         # 调试打印以查看 unique_levels 更新情况
#         print(f"Updated unique levels after removal: {unique_levels.tolist()}")
#
#         # 将当前的量化数据 Q 从[0, b1-1]映射到 [-b2 / 2, b2 / 2 - 1] 区间
#         Q_mapped = map_to_target_range(Q, 0, b1 - 1, -half_range(num_levels), half_range(num_levels) - 1)
#
#         # 保存映射后的直方图数据
#         hist_data.append(Q_mapped.clone().numpy())
#
#         # 打印出最小的误差，及其对应的level的值
#         print(f"min_level: {level_to_remove}, min_error: {total_errors[level_to_remove]}\n")
#
#     # 最后一次映射，将当前的量化数据 Q 从[0, b1-1]映射到 [-b2 / 2, b2 / 2 - 1] 区间
#     Q_mapped = map_to_target_range(Q, 0, b1 - 1, -half_range(num_levels), half_range(num_levels) - 1)
#
#     return Q_mapped, hist_data

# def dynamic_quantize_level_reduce(Q, D, num_levels):
#     unique_levels = torch.unique(Q)  # 提取 Q 中的唯一值
#     unique_levels.sort()  # 排序
#     iteration = 0  # 初始化循环计数器
#     hist_data = []  # 用于保存每一步的直方图数据
#
#     while len(unique_levels) > num_levels:
#         iteration += 1  # 循环次数加一
#
#         # 初始化移除后的误差
#         total_errors = {}
#
#         for level in unique_levels:
#             # 暂时移除当前级别的数据
#             Q_copy = Q.clone()
#
#             # 将被移除level的数据点分配给临近的level
#             level_idx = (unique_levels == level).nonzero(as_tuple=True)[0].item()
#
#             if level_idx == 0:
#                 # 如果是第一个量化级别，分配给下一个级别
#                 Q_copy[Q == level] = unique_levels[1]
#             elif level_idx == len(unique_levels) - 1:
#                 # 如果是最后一个量化级别，分配给上一个级别
#                 Q_copy[Q == level] = unique_levels[-2]
#             else:
#                 # 否则，分配到相邻的两个级别中，根据与数据的距离进行分配
#                 less_level = unique_levels[level_idx - 1].item()
#                 more_level = unique_levels[level_idx + 1].item()
#                 Q_copy[(Q == level) & (D <= level)] = less_level
#                 Q_copy[(Q == level) & (D > level)] = more_level
#
#             # 计算重新分配后的总误差
#             E = (Q_copy - D) ** 2
#             total_errors[level.item()] = E.sum().item()  # 将每个level移除后的总误差记录下来
#
#         # 找到移除后总误差最小的量化级别
#         level_to_remove = min(total_errors, key=total_errors.get)
#
#         # 重新分配数据
#         level_idx_to_remove = (unique_levels == level_to_remove).nonzero(as_tuple=True)[0].item()
#         if level_idx_to_remove == 0:
#             Q[Q == level_to_remove] = unique_levels[1]
#         elif level_idx_to_remove == len(unique_levels) - 1:
#             Q[Q == level_to_remove] = unique_levels[-2]
#         else:
#             less_level = unique_levels[level_idx_to_remove - 1].item()
#             more_level = unique_levels[level_idx_to_remove + 1].item()
#             Q[(Q == level_to_remove) & (D <= level_to_remove)] = less_level
#             Q[(Q == level_to_remove) & (D > level_to_remove)] = more_level
#
#         # 打印信息
#         print(f"Iteration {iteration}:")
#         print(f"Removed level: {level_to_remove}, Total error: {total_errors[level_to_remove]}")
#
#         # 更新唯一值
#         unique_levels = torch.unique(Q)
#         unique_levels.sort()
#
#         # 将当前的量化数据 Q 从(0,b1-1)映射到 [-((b2 - 1) / 2), ((b2 - 1) / 2)] 区间
#         Q_mapped = map_to_target_range(Q, 0, b1 - 1, -half_range_b2, half_range_b2)
#         hist_data.append(Q_mapped.clone().numpy())
#
#     return Q_mapped, hist_data


#————————————————————————————————————————————————————————————————


