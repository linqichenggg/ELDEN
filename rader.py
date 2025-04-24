import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from datetime import datetime

# 创建结果文件夹
result_folder = "三模型双盲实验比较结果"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
result_folder = f"{result_folder}_{timestamp}"
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
    print(f"已创建结果文件夹: {result_folder}")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

# 定义维度标签
dimensions = ['A. 自然流畅度', 'B. 内容真实性', 'C. 角色一致性', 
              'D. 专业可信度', 'E. 情感自然度', 'F. 总体真实感']
dim_cols = ['A', 'B', 'C', 'D', 'E', 'F']

# 加载三个实验的数据
# 第一组实验 - human评估
df_human = pd.read_csv('双盲实验结果_副本.csv')
real_human = df_human[df_human['真实id'] <= 30]
generated_human = df_human[df_human['真实id'] > 30]

# 第二组实验 - deepseek评估
df_deepseek = pd.read_csv('ratings_results/deepseek_result.csv')
# 根据ID区分真实和生成对话
real_deepseek = df_deepseek[df_deepseek['ID'].astype(str).str.startswith('1.')]
generated_deepseek = df_deepseek[df_deepseek['ID'].astype(str).str.startswith('0.')]

# 第三组实验 - o3评估
df_o3 = pd.read_csv('ratings_results/o3_result.csv')
real_o3 = df_o3[df_o3['ID'].astype(str).str.startswith('1.')]
generated_o3 = df_o3[df_o3['ID'].astype(str).str.startswith('0.')]

# 计算各个实验的均值
# 人类评估均值
human_real_means = real_human[dim_cols].mean()
human_gen_means = generated_human[dim_cols].mean()
human_diff = human_real_means - human_gen_means

# deepseek评估均值
deepseek_real_means = real_deepseek[dim_cols].mean()
deepseek_gen_means = generated_deepseek[dim_cols].mean()
deepseek_diff = deepseek_real_means - deepseek_gen_means

# o3评估均值
o3_real_means = real_o3[dim_cols].mean()
o3_gen_means = generated_o3[dim_cols].mean()
o3_diff = o3_real_means - o3_gen_means

# 创建三组实验结果的数据框
data1 = {
    '维度': dimensions,
    '真实对话均分': human_real_means.values,
    '生成对话均分': human_gen_means.values,
    '差异': (human_real_means - human_gen_means).values
}
df1 = pd.DataFrame(data1)

data2 = {
    '维度': dimensions,
    '真实对话均分': deepseek_real_means.values,
    '生成对话均分': deepseek_gen_means.values,
    '差异': (deepseek_real_means - deepseek_gen_means).values
}
df2 = pd.DataFrame(data2)

data3 = {
    '维度': dimensions,
    '真实对话均分': o3_real_means.values,
    '生成对话均分': o3_gen_means.values,
    '差异': (o3_real_means - o3_gen_means).values
}
df3 = pd.DataFrame(data3)

# 打印数据确认
print("人类评估数据:")
print(df1)
print("\nDeepSeek评估数据:")
print(df2)
print("\nO3评估数据:")
print(df3)

# 创建雷达图
plt.figure(figsize=(12, 10))
ax = plt.subplot(111, polar=True)

# 设置雷达图的角度
angles = np.linspace(0, 2*np.pi, len(dimensions), endpoint=False).tolist()
angles += angles[:1]  # 闭合雷达图

# 添加轴标签
plt.xticks(angles[:-1], dimensions, fontsize=12)

# 设置Y轴限制
ax.set_ylim(0, 5)
ax.set_yticks([1, 2, 3, 4, 5])
ax.set_yticklabels(['1', '2', '3', '4', '5'], fontsize=10)

# 绘制第一组数据 - 人类评估(蓝色)
real_values1 = human_real_means.tolist()
real_values1 += real_values1[:1]  # 闭合数据
gen_values1 = human_gen_means.tolist()
gen_values1 += gen_values1[:1]
ax.plot(angles, real_values1, 'o-', linewidth=2, color='#1f77b4', alpha=0.8, 
        label='人类评估-真实对话')
ax.plot(angles, gen_values1, 'o--', linewidth=2, color='#1f77b4', alpha=0.8, 
        label='人类评估-生成对话')

# 绘制第二组数据 - deepseek评估(红色)
real_values2 = deepseek_real_means.tolist()
real_values2 += real_values2[:1]
gen_values2 = deepseek_gen_means.tolist()
gen_values2 += gen_values2[:1]
ax.plot(angles, real_values2, 'o-', linewidth=2, color='#d62728', alpha=0.8, 
        label='DeepSeek-真实对话')
ax.plot(angles, gen_values2, 'o--', linewidth=2, color='#d62728', alpha=0.8, 
        label='DeepSeek-生成对话')

# 绘制第三组数据 - o3评估(绿色)
real_values3 = o3_real_means.tolist()
real_values3 += real_values3[:1]
gen_values3 = o3_gen_means.tolist()
gen_values3 += gen_values3[:1]
ax.plot(angles, real_values3, 'o-', linewidth=2, color='#2ca02c', alpha=0.8, 
        label='GPT-o3-真实对话')
ax.plot(angles, gen_values3, 'o--', linewidth=2, color='#2ca02c', alpha=0.8, 
        label='GPT-o3-生成对话')

# 添加图例
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=10)

# 添加标题

# 添加网格线
ax.grid(True, linestyle='-', alpha=0.3)

# 保存雷达图
plt.tight_layout()
plt.savefig(os.path.join(result_folder, '三组实验雷达图对比.png'), dpi=300, bbox_inches='tight')
plt.close()

# 创建表格汇总数据
summary_data = []
for i, dim in enumerate(dimensions):
    summary_data.append([
        dim,
        f"{human_real_means[i]:.3f} / {human_gen_means[i]:.3f} / {human_diff[i]:.3f}",
        f"{deepseek_real_means[i]:.3f} / {deepseek_gen_means[i]:.3f} / {deepseek_diff[i]:.3f}",
        f"{o3_real_means[i]:.3f} / {o3_gen_means[i]:.3f} / {o3_diff[i]:.3f}"
    ])

# 创建数据汇总表格
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')
table_data = [['维度', '人类评估 (真/生/差)', 'DeepSeek (真/生/差)', 'GPT-o3 (真/生/差)']]
table_data.extend(summary_data)
table = ax.table(cellText=table_data, cellLoc='center', loc='center', 
                 colWidths=[0.25, 0.25, 0.25, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)

# 保存表格
plt.savefig(os.path.join(result_folder, '三组实验数据表格.png'), dpi=300, bbox_inches='tight')
plt.close()

# 创建差异值对比柱状图
fig, ax = plt.subplots(figsize=(14, 8))
ind = np.arange(len(dimensions))
width = 0.25

# 转换为数组以便绘图
diff1 = (human_real_means - human_gen_means).values
diff2 = (deepseek_real_means - deepseek_gen_means).values
diff3 = (o3_real_means - o3_gen_means).values

bar1 = ax.bar(ind - width, diff1, width, label='人类评估差异', color='#1f77b4', alpha=0.8)
bar2 = ax.bar(ind, diff2, width, label='DeepSeek差异', color='#d62728', alpha=0.8)
bar3 = ax.bar(ind + width, diff3, width, label='GPT-o3差异', color='#2ca02c', alpha=0.8)

# 为0值添加一条参考线
ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

# 添加标签和图例
ax.set_ylabel('差异值(真实对话-生成对话)', fontsize=12)
ax.set_xlabel('维度', fontsize=12)
ax.set_xticks(ind)
ax.set_xticklabels(dimensions)
ax.legend()

# 添加网格线
ax.grid(True, linestyle='--', alpha=0.3, axis='y')

# 在每个柱子上添加数值标签
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        if height < 0:
            y_pos = height - 0.01
        else:
            y_pos = height + 0.01
        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{height:.3f}', ha='center', va='bottom', rotation=0, fontsize=8)

add_labels(bar1)
add_labels(bar2)
add_labels(bar3)

# 保存差异图
plt.tight_layout()
plt.savefig(os.path.join(result_folder, '三组实验差异值对比.png'), dpi=600, bbox_inches='tight')
plt.close()

# 计算总体平均分并创建平均分对比图
human_real_avg = human_real_means.mean()
human_gen_avg = human_gen_means.mean()
deepseek_real_avg = deepseek_real_means.mean()
deepseek_gen_avg = deepseek_gen_means.mean()
o3_real_avg = o3_real_means.mean()
o3_gen_avg = o3_gen_means.mean()

# 创建平均分组图
fig, ax = plt.subplots(figsize=(10, 6))
X = np.arange(3)
width = 0.4

# 绘制真实和生成对话的平均分
ax.bar(X - width/2, [human_real_avg, deepseek_real_avg, o3_real_avg], width, label='真实对话', color='royalblue')
ax.bar(X + width/2, [human_gen_avg, deepseek_gen_avg, o3_gen_avg], width, label='生成对话', color='lightcoral')

# 设置x轴标签和图例
ax.set_xticks(X)
ax.set_xticklabels(['人类评估', 'DeepSeek', 'GPT-o3'])
ax.set_ylabel('平均分', fontsize=12)
ax.legend()

# 在柱子上添加具体分数
for i, v in enumerate([human_real_avg, deepseek_real_avg, o3_real_avg]):
    ax.text(i - width/2, v + 0.05, f'{v:.3f}', ha='center', fontsize=9)
    
for i, v in enumerate([human_gen_avg, deepseek_gen_avg, o3_gen_avg]):
    ax.text(i + width/2, v + 0.05, f'{v:.3f}', ha='center', fontsize=9)

# 添加网格线
ax.grid(True, linestyle='--', alpha=0.3, axis='y')
ax.set_ylim(0, 5)

# 保存平均分对比图
plt.tight_layout()
plt.savefig(os.path.join(result_folder, '三组实验平均分对比.png'), dpi=300)
plt.close()

print(f"已生成三组实验对比图表，保存在: {result_folder}")