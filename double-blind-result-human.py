import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib as mpl
import platform
import os
from datetime import datetime

# 创建结果文件夹
result_folder = "双盲实验分析结果"
# 添加时间戳以避免覆盖
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
result_folder = f"{result_folder}_{timestamp}"

# 确保文件夹存在
if not os.path.exists(result_folder):
    os.makedirs(result_folder)
    print(f"已创建结果文件夹: {result_folder}")

# 图表子文件夹
img_folder = os.path.join(result_folder, "图表")
if not os.path.exists(img_folder):
    os.makedirs(img_folder)

# 检测操作系统并设置合适的中文字体
system = platform.system()
if system == 'Windows':
    font_name = 'Microsoft YaHei'
elif system == 'Darwin':  # macOS
    font_name = 'PingFang SC'  # macOS常见中文字体
else:  # Linux
    font_name = 'WenQuanYi Micro Hei'  # Linux常见中文字体

# 设置matplotlib参数
plt.rcParams['font.sans-serif'] = [font_name, 'Arial Unicode MS', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

# 加载数据
df = pd.read_csv('双盲实验结果_副本.csv')

# 根据真实ID分组
real_data = df[df['真实id'] <= 30]
generated_data = df[df['真实id'] > 30]

# 定义维度标签
dimensions = ['A. 自然流畅度', 'B. 内容真实性', 'C. 角色一致性', 
              'D. 专业可信度', 'E. 情感自然度', 'F. 总体真实感']
dim_cols = ['A', 'B', 'C', 'D', 'E', 'F']

# 1. 描述性统计
real_stats = real_data[dim_cols].describe()
generated_stats = generated_data[dim_cols].describe()

print("真实对话统计:")
print(real_stats)
print("\n生成对话统计:")
print(generated_stats)

# 2. 统计检验
results = {}
for col in dim_cols:
    t_stat, p_val = stats.ttest_ind(real_data[col], generated_data[col], equal_var=False)
    # 计算Cohen's d效应量
    cohens_d = (real_data[col].mean() - generated_data[col].mean()) / np.sqrt(
        (real_data[col].std()**2 + generated_data[col].std()**2) / 2)
    
    results[col] = {
        '真实组均值': real_data[col].mean(),
        '生成组均值': generated_data[col].mean(),
        '差异': real_data[col].mean() - generated_data[col].mean(),
        't统计量': t_stat, 
        'p值': p_val,
        '效应量': cohens_d,
        '是否显著': p_val < 0.05
    }

results_df = pd.DataFrame(results).T
print("\n统计检验结果:")
print(results_df)

# 3. 可视化分析
# 3.1 箱线图比较
plt.figure(figsize=(15, 10))
for i, col in enumerate(dim_cols, 1):
    plt.subplot(2, 3, i)
    data = [real_data[col], generated_data[col]]
    # 使用tick_labels代替labels参数
    plt.boxplot(data, tick_labels=['真实对话', '生成对话'])
    plt.title(f'{dimensions[i-1]}')
    plt.grid(True, linestyle='--', alpha=0.7)
    
plt.tight_layout()
plt.suptitle('各维度评分箱线图比较', fontsize=16, y=1.02)
plt.savefig(os.path.join(img_folder, '箱线图比较.png'), dpi=300, bbox_inches='tight')
plt.close()

# 3.2 平均分柱状图
means_real = real_data[dim_cols].mean()
means_generated = generated_data[dim_cols].mean()

plt.figure(figsize=(12, 8))
x = np.arange(len(dim_cols))
width = 0.35

plt.bar(x - width/2, means_real, width, label='真实对话')
plt.bar(x + width/2, means_generated, width, label='生成对话')

plt.xlabel('评分维度')
plt.ylabel('平均分')
plt.title('真实对话 vs 生成对话 各维度平均分')
plt.xticks(x, dimensions, rotation=15)
plt.grid(True, linestyle='--', alpha=0.3, axis='y')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(img_folder, '平均分柱状图.png'), dpi=300)
plt.close()

# 3.3 雷达图
plt.figure(figsize=(10, 10))
angles = np.linspace(0, 2*np.pi, len(dim_cols), endpoint=False).tolist()
angles += angles[:1]  # 闭合雷达图

means_real_list = means_real.tolist()
means_real_list += means_real_list[:1]  # 闭合数据

means_generated_list = means_generated.tolist()
means_generated_list += means_generated_list[:1]  # 闭合数据

ax = plt.subplot(111, polar=True)
ax.plot(angles, means_real_list, 'o-', linewidth=2, label='真实对话')
ax.fill(angles, means_real_list, alpha=0.25)
ax.plot(angles, means_generated_list, 'o-', linewidth=2, label='生成对话')
ax.fill(angles, means_generated_list, alpha=0.25)

ax.set_thetagrids(np.degrees(angles[:-1]), dimensions)
ax.set_ylim(0, 5)
ax.grid(True)
plt.legend(loc='upper right')
plt.title('真实对话 vs 生成对话 多维度评分')
plt.tight_layout()
plt.savefig(os.path.join(img_folder, '雷达图.png'), dpi=300)
plt.close()

# 3.4 分组对比图 - 小提琴图
plt.figure(figsize=(14, 8))
df_melted = pd.melt(df, id_vars=['id', '真实id'], value_vars=dim_cols, 
                    var_name='维度', value_name='评分')
df_melted['数据类型'] = df_melted['真实id'].apply(lambda x: '真实对话' if x <= 30 else '生成对话')

sns.violinplot(x='维度', y='评分', hue='数据类型', data=df_melted, split=True, inner='quart')
plt.title('真实对话与生成对话各维度评分分布')
plt.grid(True, linestyle='--', alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(img_folder, '小提琴图.png'), dpi=300)
plt.close()

# 3.5 散点图矩阵 - 各维度相关性可视化
plt.figure(figsize=(14, 12))
axes = pd.plotting.scatter_matrix(df[dim_cols], alpha=0.8, figsize=(14, 12), diagonal='kde')
for i in range(len(dim_cols)):
    for j in range(len(dim_cols)):
        if i != j:
            # 为每个散点添加分组颜色
            ax = axes[i, j]
            ax.scatter(df[df['真实id'] <= 30][dim_cols[j]], 
                       df[df['真实id'] <= 30][dim_cols[i]], 
                       color='blue', alpha=0.5, s=30, label='真实对话')
            ax.scatter(df[df['真实id'] > 30][dim_cols[j]], 
                       df[df['真实id'] > 30][dim_cols[i]], 
                       color='red', alpha=0.5, s=30, label='生成对话')
            
            # 只在第一个图添加图例
            if i == 1 and j == 0:
                ax.legend()

plt.suptitle('各维度散点图矩阵', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(img_folder, '散点图矩阵.png'), dpi=300, bbox_inches='tight')
plt.close()

# 4. 平均分比较
avg_real = real_data['平均分'].mean()
avg_generated = generated_data['平均分'].mean()
t_stat, p_val = stats.ttest_ind(real_data['平均分'], generated_data['平均分'], equal_var=False)
cohens_d = (avg_real - avg_generated) / np.sqrt(
    (real_data['平均分'].std()**2 + generated_data['平均分'].std()**2) / 2)

print("\n总平均分比较:")
print(f"真实对话平均分: {avg_real:.3f}")
print(f"生成对话平均分: {avg_generated:.3f}")
print(f"差异: {avg_real - avg_generated:.3f}")
print(f"t统计量: {t_stat:.3f}, p值: {p_val:.5f}")
print(f"效应量(Cohen's d): {cohens_d:.3f}")
print(f"差异是否显著: {p_val < 0.05}")

# 可视化平均分比较
plt.figure(figsize=(8, 6))
plt.bar(['真实对话', '生成对话'], [avg_real, avg_generated], color=['blue', 'orange'])
plt.ylabel('总平均分')
plt.title('真实对话 vs 生成对话 总平均分比较')
plt.grid(True, linestyle='--', alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(img_folder, '总平均分比较.png'), dpi=300)
plt.close()

# 5. 详细结果表格
summary = pd.DataFrame({
    '维度': dimensions,
    '真实对话均分': means_real.values,
    '生成对话均分': means_generated.values,
    '差异': means_real.values - means_generated.values,
    'p值': [results[col]['p值'] for col in dim_cols],
    '是否显著': [results[col]['是否显著'] for col in dim_cols],
    '效应量': [results[col]['效应量'] for col in dim_cols]
})

print("\n详细结果表格:")
print(summary.sort_values('差异', ascending=False))

# 6. 总体真实感(F维度)的详细分析
print("\n总体真实感(F维度)分析:")
print(f"真实对话得分: {real_data['F'].mean():.3f} ± {real_data['F'].std():.3f}")
print(f"生成对话得分: {generated_data['F'].mean():.3f} ± {generated_data['F'].std():.3f}")

# F维度直方图比较
plt.figure(figsize=(10, 6))
plt.hist(real_data['F'], alpha=0.7, bins=10, label='真实对话', color='blue')
plt.hist(generated_data['F'], alpha=0.7, bins=10, label='生成对话', color='orange')
plt.xlabel('总体真实感评分')
plt.ylabel('频数')
plt.title('F维度(总体真实感)评分分布')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(img_folder, 'F维度分布.png'), dpi=300)
plt.close()

# 7. 相关性分析 - 各维度之间的相关性
print("\n维度间的相关性分析:")
correlation_matrix = df[dim_cols].corr()
print(correlation_matrix)

# 保存相关性热图
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('各评分维度间的相关性')
plt.tight_layout()
plt.savefig(os.path.join(img_folder, '相关性热图.png'), dpi=300)
plt.close()

# 8. 分组相关性分析
real_corr = real_data[dim_cols].corr()
generated_corr = generated_data[dim_cols].corr()

# 真实组相关性热图
plt.figure(figsize=(10, 8))
sns.heatmap(real_corr, annot=True, cmap='Blues', fmt='.2f')
plt.title('真实对话组 - 各评分维度间的相关性')
plt.tight_layout()
plt.savefig(os.path.join(img_folder, '真实组相关性热图.png'), dpi=300)
plt.close()

# 生成组相关性热图
plt.figure(figsize=(10, 8))
sns.heatmap(generated_corr, annot=True, cmap='Oranges', fmt='.2f')
plt.title('生成对话组 - 各评分维度间的相关性')
plt.tight_layout()
plt.savefig(os.path.join(img_folder, '生成组相关性热图.png'), dpi=300)
plt.close()

# 9. 保存结果到Excel文件夹
excel_path = os.path.join(result_folder, "双盲实验分析结果.xlsx")
with pd.ExcelWriter(excel_path) as writer:
    real_stats.to_excel(writer, sheet_name='真实对话统计')
    generated_stats.to_excel(writer, sheet_name='生成对话统计')
    results_df.to_excel(writer, sheet_name='统计检验结果')
    summary.to_excel(writer, sheet_name='详细比较结果')
    correlation_matrix.to_excel(writer, sheet_name='整体相关性')
    real_corr.to_excel(writer, sheet_name='真实组相关性')
    generated_corr.to_excel(writer, sheet_name='生成组相关性')

# 10. 生成摘要报告
report_path = os.path.join(result_folder, "分析报告.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("# 双盲实验分析报告\n\n")
    f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("## 1. 总体比较\n\n")
    f.write(f"真实对话平均分: {avg_real:.3f}\n")
    f.write(f"生成对话平均分: {avg_generated:.3f}\n")
    f.write(f"差异: {avg_real - avg_generated:.3f}\n")
    f.write(f"t统计量: {t_stat:.3f}, p值: {p_val:.5f}\n")
    f.write(f"效应量(Cohen's d): {cohens_d:.3f}\n")
    f.write(f"差异是否显著: {p_val < 0.05}\n\n")
    
    f.write("## 2. 各维度比较\n\n")
    for i, col in enumerate(dim_cols):
        result = results[col]
        f.write(f"### {dimensions[i]}\n")
        f.write(f"真实对话均分: {result['真实组均值']:.3f}\n")
        f.write(f"生成对话均分: {result['生成组均值']:.3f}\n")
        f.write(f"差异: {result['差异']:.3f}\n")
        f.write(f"p值: {result['p值']:.5f} (差异{'' if result['是否显著'] else '不'}显著)\n")
        f.write(f"效应量: {result['效应量']:.3f}\n\n")
    
    f.write("## 3. 总体真实感(F维度)详细分析\n\n")
    f.write(f"真实对话得分: {real_data['F'].mean():.3f} ± {real_data['F'].std():.3f}\n")
    f.write(f"生成对话得分: {generated_data['F'].mean():.3f} ± {generated_data['F'].std():.3f}\n\n")
    
    f.write("## 4. 结论\n\n")
    sig_dims = [dimensions[i] for i, col in enumerate(dim_cols) if results[col]['是否显著']]
    if sig_dims:
        f.write(f"在以下维度上，真实对话和生成对话存在显著差异: {', '.join(sig_dims)}\n\n")
    else:
        f.write("在所有维度上，真实对话和生成对话均无显著差异。\n\n")
    
    largest_diff_dim = dimensions[np.argmax(np.abs([results[col]['差异'] for col in dim_cols]))]
    f.write(f"最大差异出现在 {largest_diff_dim} 维度。\n\n")
    
    f.write("真实对话和生成对话的总体评分差距是否显著取决于p值。如果p<0.05，则差距显著。\n")

print(f"\n分析完成，所有结果已保存到 {result_folder} 文件夹")
print(f"图表文件位于: {img_folder}")
print(f"Excel结果文件: {excel_path}")
print(f"分析报告: {report_path}")