import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 定义节点的位置
input_pos = [(0.2, 0.8), (0.4, 0.8), (0.6, 0.8), (0.8, 0.8)]
hidden_pos = [(0.2, 0.5), (0.4, 0.5), (0.6, 0.5), (0.8, 0.5), (0.5, 0.3)]  # 五个隐藏层节点
output_pos = [(0.5, 0.2)]

# 创建图形和轴
fig, ax = plt.subplots()

# 定义颜色
colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange']  # 增加一个颜色用于隐藏层

# 绘制输入层节点
for i, pos in enumerate(input_pos):
    circle = plt.Circle(pos, 0.1, color=colors[i], fill=True, edgecolor='none')
    ax.add_patch(circle)
    plt.text(pos[0], pos[1], 'X' if i == 0 else 'Y' if i == 1 else 'Z' if i == 2 else 'W', ha='center', va='center')

# 绘制隐藏层节点
for i, pos in enumerate(hidden_pos):
    circle = plt.Circle(pos, 0.1, color='orange', fill=True, edgecolor='none')  # 隐藏层节点颜色统一
    ax.add_patch(circle)

# 绘制输出层节点
for i, pos in enumerate(output_pos):
    circle = plt.Circle(pos, 0.1, color=colors[1], fill=True, edgecolor='none')  # 输出层颜色与输入层Y相同
    ax.add_patch(circle)
    plt.text(pos[0], pos[1], 'Y', ha='center', va='center')

# 设置图形属性
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.axis('off')  # 关闭坐标轴

# 显示图形
plt.show()