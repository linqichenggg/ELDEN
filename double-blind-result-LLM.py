import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from datetime import datetime

# 创建结果文件夹
result_folder = "大模型评估结果分析"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
result_folder = f"{result_folder}_{timestamp}"

if not os.path.exists(result_folder):
    os.makedirs(result_folder)
    print(f"已创建结果文件夹: {result_folder}")

# 图表子文件夹
img_folder = os.path.join(result_folder, "图表")
if not os.path.exists(img_folder):
    os.makedirs(img_folder)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'

# 加载数据
df = pd.read_csv('ratings_results/o3_result.csv', sep=',')

# 检查列名是否存在空格
print(df.columns.tolist())

# 确认F列是否存在
if 'F' in df.columns:
    print("F列存在，前5个值:", df['F'].head())
else:
    print("F列不存在，实际列名为:", df.columns)

# 区分真实对话和生成对话组
# 根据ID判断：ID以1.开头的是真实对话，ID以0.开头的是生成对话
real_data = df[df['ID'].astype(str).str.startswith('1.')]
generated_data = df[df['ID'].astype(str).str.startswith('0.')]

# 计算每组样本数
n_real = len(real_data)
n_generated = len(generated_data)
print(f"真实对话样本数: {n_real}")
print(f"生成对话样本数: {n_generated}")

# 定义维度标签和列名
dimensions = ['A. 自然流畅度', 'B. 内容真实性', 'C. 角色一致性', 
              'D. 专业可信度', 'E. 情感自然度', 'F. 总体真实感']
dim_cols = ['A', 'B', 'C', 'D', 'E', 'F']

# 1. 描述性统计
real_stats = real_data[dim_cols].describe()
generated_stats = generated_data[dim_cols].describe()

print("\n真实对话统计:")
print(real_stats)
print("\n生成对话统计:")
print(generated_stats)

# 计算每个样本的平均分
real_data['平均分'] = real_data[dim_cols].mean(axis=1)
generated_data['平均分'] = generated_data[dim_cols].mean(axis=1)

# 2. 统计检验和效应量计算
results = {}
for col in dim_cols:
    # 进行t检验
    t_stat, p_val = stats.ttest_ind(real_data[col], generated_data[col], equal_var=False)
    
    # 计算Cohen's d效应量
    cohens_d = (real_data[col].mean() - generated_data[col].mean()) / np.sqrt(
        (real_data[col].std()**2 + generated_data[col].std()**2) / 2)
    
    # 效应量分类
    if abs(cohens_d) < 0.2:
        effect_size_cat = "可忽略"
    elif abs(cohens_d) < 0.5:
        effect_size_cat = "小效应"
    elif abs(cohens_d) < 0.8:
        effect_size_cat = "中等效应"
    else:
        effect_size_cat = "大效应"
    
    # 保存结果
    results[col] = {
        '真实组均值': real_data[col].mean(),
        '生成组均值': generated_data[col].mean(),
        '差异': real_data[col].mean() - generated_data[col].mean(),
        't统计量': t_stat, 
        'p值': p_val,
        '效应量': cohens_d,
        '效应量类别': effect_size_cat,
        '是否显著': p_val < 0.05
    }

# 总平均分检验
t_stat, p_val = stats.ttest_ind(real_data['平均分'], generated_data['平均分'], equal_var=False)
cohens_d = (real_data['平均分'].mean() - generated_data['平均分'].mean()) / np.sqrt(
    (real_data['平均分'].std()**2 + generated_data['平均分'].std()**2) / 2)

if abs(cohens_d) < 0.2:
    effect_size_cat = "可忽略"
elif abs(cohens_d) < 0.5:
    effect_size_cat = "小效应"
elif abs(cohens_d) < 0.8:
    effect_size_cat = "中等效应"
else:
    effect_size_cat = "大效应"

results['平均分'] = {
    '真实组均值': real_data['平均分'].mean(),
    '生成组均值': generated_data['平均分'].mean(),
    '差异': real_data['平均分'].mean() - generated_data['平均分'].mean(),
    't统计量': t_stat, 
    'p值': p_val,
    '效应量': cohens_d,
    '效应量类别': effect_size_cat,
    '是否显著': p_val < 0.05
}

# 将结果转换为DataFrame
results_df = pd.DataFrame(results).T

# 打印结果
print("\n统计检验结果:")
print(results_df)

# 3. 多重比较校正
# 使用Bonferroni校正
alpha = 0.05
bonferroni_alpha = alpha / len(dim_cols)

print(f"\nBonferroni校正后的显著性水平: {bonferroni_alpha:.5f}")
for col in dim_cols:
    p_val = results[col]['p值']
    print(f"{col} 维度p值: {p_val:.5f}, {'显著' if p_val < bonferroni_alpha else '不显著'}")

# 4. 可视化分析
# 4.1 平均分柱状图
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

# 4.2 箱线图
plt.figure(figsize=(15, 10))
for i, col in enumerate(dim_cols, 1):
    plt.subplot(2, 3, i)
    data = [real_data[col], generated_data[col]]
    plt.boxplot(data, labels=['真实对话', '生成对话'])
    plt.title(f'{dimensions[i-1]}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(1, 5)  # 统一Y轴范围
    
plt.tight_layout()
plt.suptitle('各维度评分箱线图比较', fontsize=16, y=1.02)
plt.savefig(os.path.join(img_folder, '箱线图比较.png'), dpi=300, bbox_inches='tight')
plt.close()

# 4.3 雷达图
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
ax.set_ylim(1, 5)
ax.grid(True)
plt.legend(loc='upper right')
plt.title('真实对话 vs 生成对话 多维度评分')
plt.tight_layout()
plt.savefig(os.path.join(img_folder, '雷达图.png'), dpi=300)
plt.close()

# 4.4 小提琴图
plt.figure(figsize=(14, 8))
df_melted = pd.melt(df, id_vars=['ID'], value_vars=dim_cols, 
                    var_name='维度', value_name='评分')
df_melted['数据类型'] = df_melted['ID'].astype(str).str.startswith('1.').map({True: '真实对话', False: '生成对话'})

sns.violinplot(x='维度', y='评分', hue='数据类型', data=df_melted, split=True, inner='quart')
plt.title('真实对话与生成对话各维度评分分布')
plt.grid(True, linestyle='--', alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(img_folder, '小提琴图.png'), dpi=300)
plt.close()

# 4.5 热图
# 相关性热图
plt.figure(figsize=(10, 8))
corr_matrix = df[dim_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('各维度评分相关性')
plt.tight_layout()
plt.savefig(os.path.join(img_folder, '相关性热图.png'), dpi=300)
plt.close()

# 5. 总体真实感(F维度)单独分析
plt.figure(figsize=(10, 6))
plt.hist(real_data['F'], alpha=0.7, bins=np.arange(1.5, 5.5, 0.5), label='真实对话', color='blue')
plt.hist(generated_data['F'], alpha=0.7, bins=np.arange(1.5, 5.5, 0.5), label='生成对话', color='orange')
plt.xlabel('总体真实感评分')
plt.ylabel('频数')
plt.title('F维度(总体真实感)评分分布')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(img_folder, 'F维度分布.png'), dpi=300)
plt.close()

# 6. 生成详细结果表格
summary = pd.DataFrame({
    '维度': dimensions + ['总平均分'],
    '真实对话均分': [results[col]['真实组均值'] for col in dim_cols] + [results['平均分']['真实组均值']],
    '生成对话均分': [results[col]['生成组均值'] for col in dim_cols] + [results['平均分']['生成组均值']],
    '差异': [results[col]['差异'] for col in dim_cols] + [results['平均分']['差异']],
    'p值': [results[col]['p值'] for col in dim_cols] + [results['平均分']['p值']],
    'p值校正(Bonferroni)': [results[col]['p值'] * len(dim_cols) for col in dim_cols] + [np.nan],
    '是否显著': [results[col]['是否显著'] for col in dim_cols] + [results['平均分']['是否显著']],
    '效应量': [results[col]['效应量'] for col in dim_cols] + [results['平均分']['效应量']],
    '效应量类别': [results[col]['效应量类别'] for col in dim_cols] + [results['平均分']['效应量类别']]
})

print("\n详细结果表格:")
print(summary)

# 7. 保存结果到Excel
excel_path = os.path.join(result_folder, "大模型评估分析结果.xlsx")
with pd.ExcelWriter(excel_path) as writer:
    real_stats.to_excel(writer, sheet_name='真实对话统计')
    generated_stats.to_excel(writer, sheet_name='生成对话统计')
    results_df.to_excel(writer, sheet_name='统计检验结果')
    summary.to_excel(writer, sheet_name='详细比较结果')
    corr_matrix.to_excel(writer, sheet_name='维度相关性')

# 8. 生成摘要报告
report_path = os.path.join(result_folder, "分析报告.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("# 大模型评估双盲实验分析报告\n\n")
    f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    f.write("## 1. 总体比较\n\n")
    f.write(f"真实对话平均分: {results['平均分']['真实组均值']:.3f}\n")
    f.write(f"生成对话平均分: {results['平均分']['生成组均值']:.3f}\n")
    f.write(f"差异: {results['平均分']['差异']:.3f}\n")
    f.write(f"t统计量: {results['平均分']['t统计量']:.3f}, p值: {results['平均分']['p值']:.5f}\n")
    f.write(f"效应量(Cohen's d): {results['平均分']['效应量']:.3f} ({results['平均分']['效应量类别']})\n")
    f.write(f"差异是否显著: {results['平均分']['是否显著']}\n\n")
    
    f.write("## 2. 各维度比较\n\n")
    for i, col in enumerate(dim_cols):
        result = results[col]
        f.write(f"### {dimensions[i]}\n")
        f.write(f"真实对话均分: {result['真实组均值']:.3f}\n")
        f.write(f"生成对话均分: {result['生成组均值']:.3f}\n")
        f.write(f"差异: {result['差异']:.3f}\n")
        f.write(f"p值: {result['p值']:.5f} (差异{'' if result['是否显著'] else '不'}显著)\n")
        f.write(f"Bonferroni校正后p值: {result['p值'] * len(dim_cols):.5f}\n")
        f.write(f"效应量: {result['效应量']:.3f} ({result['效应量类别']})\n\n")
    
    f.write("## 3. 多重比较校正\n\n")
    f.write(f"Bonferroni校正后的显著性水平: α' = α/k = 0.05/{len(dim_cols)} = {bonferroni_alpha:.5f}\n")
    sig_dims_bonf = [dimensions[i] for i, col in enumerate(dim_cols) if results[col]['p值'] < bonferroni_alpha]
    if sig_dims_bonf:
        f.write(f"校正后仍显著的维度: {', '.join(sig_dims_bonf)}\n\n")
    else:
        f.write("校正后所有维度均不显著\n\n")
    
    f.write("## 4. 总体真实感(F维度)详细分析\n\n")
    f.write(f"真实对话得分: {real_data['F'].mean():.3f} ± {real_data['F'].std():.3f}\n")
    f.write(f"生成对话得分: {generated_data['F'].mean():.3f} ± {generated_data['F'].std():.3f}\n\n")
    
    f.write("## 5. 结论\n\n")
    sig_dims = [dimensions[i] for i, col in enumerate(dim_cols) if results[col]['是否显著']]
    if sig_dims:
        f.write(f"在以下维度上，真实对话和生成对话存在显著差异: {', '.join(sig_dims)}\n\n")
    else:
        f.write("在不校正的情况下，所有维度上真实对话和生成对话均无显著差异。\n\n")
    
    largest_diff_dim = dimensions[np.argmax(np.abs([results[col]['差异'] for col in dim_cols]))]
    f.write(f"最大差异出现在 {largest_diff_dim} 维度。\n\n")
    
    # 效应量解释
    f.write("## 6. 效应量解释\n\n")
    f.write("| 效应量(d)范围 | 效应程度 | 实际意义 |\n")
    f.write("|:--------------|:--------:|:---------|\n")
    f.write("| 0.00-0.20 | 可忽略效应 | 差异极小，几乎无实际意义 |\n")
    f.write("| 0.20-0.50 | 小效应 | 存在小幅差异，但实际影响有限 |\n")
    f.write("| 0.50-0.80 | 中等效应 | 差异明显，具有实际意义 |\n")
    f.write("| >0.80 | 大效应 | 差异显著，具有重要实际意义 |\n\n")
    
    f.write("## 7. 总体评价\n\n")
    f.write("本分析比较了大模型评估的真实对话和生成对话在六个维度上的评分差异。\n")
    if results['平均分']['是否显著']:
        f.write(f"真实对话和生成对话的总体评分存在显著差异(p={results['平均分']['p值']:.5f})，")
        if results['平均分']['差异'] > 0:
            f.write("真实对话的评分显著高于生成对话。\n")
        else:
            f.write("生成对话的评分显著高于真实对话。\n")
    else:
        f.write(f"真实对话和生成对话的总体评分差异不显著(p={results['平均分']['p值']:.5f})。\n")
    
    large_effect_dims = [dimensions[i] for i, col in enumerate(dim_cols) if abs(results[col]['效应量']) >= 0.8]
    if large_effect_dims:
        f.write(f"\n在以下维度存在大效应差异: {', '.join(large_effect_dims)}\n")

print(f"\n分析完成，所有结果已保存到 {result_folder} 文件夹")
print(f"图表文件位于: {img_folder}")
print(f"Excel结果文件: {excel_path}")
print(f"分析报告: {report_path}")