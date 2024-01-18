import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取CSV文件
df = pd.read_csv('plots/speed.csv', header=None)

# 转置DataFrame以便每一列代表一个坐标轴的数据
df = df.T

# 第一列是x，第二列是y
x = df[0]
y = pd.to_numeric(df[1], errors='coerce')  # 将y列转换为数值类型，非数值转为NaN


# 创建一个新的图形和轴对象
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

# 生成颜色数组，根据y值的大小分配颜色，使用颜色映射(colormap)
colors = y - y.min()  # 归一化y值
colors = 5 * (colors / colors.max())  # 将y值缩放到0-1范围
cmap = plt.colormaps.get_cmap('Blues')  # 选择一个颜色映射
point_colors = cmap(colors)  # 应用颜色映射

# 绘制散点图，每个点颜色不同
sc = ax.scatter(x, y, alpha=0.7, c=point_colors, edgecolors='w', s=50, linewidth=0.6)



# 调整x轴刻度标签的字体大小以避免重叠
ax.tick_params(axis='x', labelsize=7)

# 添加标题和轴标签，使用合适的字体大小
ax.set_title('speed of llms', fontsize=16, fontweight='bold')
ax.set_xlabel('models', fontsize=12)
ax.set_ylabel('speed(tokens/s)', fontsize=14)

# 设置网格线（可选）
ax.grid(True, linestyle='--', linewidth=0.5)

# 保存图形
plt.savefig('plots/scatter_plot.png', format='png', bbox_inches='tight')

# 显示图形
plt.show()
