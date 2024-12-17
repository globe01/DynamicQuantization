import os
import pandas as pd

# 文件夹路径
folder_path = './quantization_results'

# 初始化存储数据的列表
data = []

# 遍历文件夹中的所有文件
for filename in sorted(os.listdir(folder_path), key=lambda x: [int(part) if part.isdigit() else part for part in x.replace('_errors.txt', '').split('_')]):
    if filename.endswith('_errors.txt'):
        # 获取文件完整路径
        file_path = os.path.join(folder_path, filename)

        # 提取 layer 名称
        layer_name = filename.replace('_errors.txt', '').replace('_', '.')

        # 读取文件内容
        with open(file_path, 'r') as file:
            lines = file.readlines()
            if len(lines) >= 2:
                # 提取 Dynamic MSE 和 Uniform MSE
                dynamic_mse, uniform_mse = lines[1].strip().split()

                # 将数据存储到列表中
                data.append([layer_name, float(dynamic_mse), float(uniform_mse)])

# 创建 Pandas DataFrame
df = pd.DataFrame(data, columns=['Layer', 'Dynamic MSE', 'Uniform MSE'])

# 保存为 Excel 文件
output_file = 'quantization_results.xlsx'
df.to_excel(output_file, index=False)

print(f"数据已成功保存到 {output_file}")
