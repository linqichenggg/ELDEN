import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import json
import os
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
import matplotlib as mpl
import matplotlib.font_manager as fm
from scipy.stats import chi2_contingency
from sklearn.tree import DecisionTreeClassifier, export_text

# 使用更通用的字体设置方法
try:
    # 尝试使用系统自带的中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft YaHei', 'SimHei', 'STHeiti', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 测试中文显示
    mpl.rc('font', family='sans-serif')
    fig = plt.figure(figsize=(1, 1))
    plt.text(0.5, 0.5, '测试中文', ha='center', va='center')
    plt.close(fig)
    print("成功启用中文字体")
except:
    print("无法找到中文字体，使用默认设置")
    # 如果找不到中文字体，则不使用中文标签
    use_chinese_labels = False

# 创建输出目录
os.makedirs("analysis_results/goal1.3", exist_ok=True)

###########################################
# 第1步: 数据加载与预处理
###########################################

# 加载weibo用户数据
weibo_df = pd.read_csv("/Users/lqcmacmini/Desktop/weibo_users_extended.csv")

# 加载代理人信念数据
with open("output/run-1/final_3-agent-beliefs.json", "r", encoding="utf-8") as f:
    belief_data = json.load(f)

# 创建用户名到特质的映射
user_traits = {}
for _, row in weibo_df.iterrows():
    user_traits[row['用户名']] = {
        'openness': int(row['开放性']),
        'conscientiousness': int(row['尽责性']),
        'extraversion': int(row['外向性']),
        'agreeableness': int(row['宜人性']),
        'neuroticism': int(row['神经质']),
        'education': int(row['教育背景']),
        'self_description': row['自我描述'] if '自我描述' in weibo_df.columns else ""
    }

# 创建特质名称映射（用于中文显示）
trait_names = {
    'openness': '开放性',
    'conscientiousness': '尽责性',
    'extraversion': '外向性',
    'agreeableness': '宜人性',
    'neuroticism': '神经质'
}

###########################################
# 第2步: 数据分析与特征提取
###########################################

# 提取信念变化指标
analysis_data = []

for agent in belief_data["agents"]:
    agent_id = agent["id"]
    agent_name = agent["name"]
    beliefs = agent["belief_history"]
    
    # 跳过无法匹配特质数据的代理人
    if agent_name not in user_traits:
        print(f"警告：找不到代理人 '{agent_name}' 的特质数据，已跳过")
        continue
    
    # 计算信念变化次数
    belief_changes = 0
    for i in range(1, len(beliefs)):
        if beliefs[i] != beliefs[i-1]:
            belief_changes += 1
    
    # 首次感染时间（从0变为1的第一个步骤）
    infection_time = None
    for i in range(1, len(beliefs)):
        if beliefs[i-1] == 0 and beliefs[i] == 1:
            infection_time = i
            break
    
    # 首次恢复时间（从1变为0的第一个步骤）
    recovery_time = None
    for i in range(1, len(beliefs)):
        if beliefs[i-1] == 1 and beliefs[i] == 0:
            recovery_time = i
            break
    
    # 总感染持续时间（信念为1的步骤总数）
    total_infected_time = sum(beliefs)
    
    # 最长连续感染时间
    max_consecutive_infected = 0
    current_consecutive = 0
    for belief in beliefs:
        if belief == 1:
            current_consecutive += 1
            max_consecutive_infected = max(max_consecutive_infected, current_consecutive)
        else:
            current_consecutive = 0
    
    # 抵抗能力指数（初始不信，保持不信的时间比例）
    resistance_index = 0
    if beliefs[0] == 0:
        # 如果从未感染，则抵抗力指数为1
        if infection_time is None:
            resistance_index = 1
        else:
            # 否则，计算感染前保持不信的时间比例
            resistance_index = infection_time / len(beliefs)
    
    # 恢复能力指数（如果曾经感染，从感染到恢复的速度）
    recovery_index = 0
    if infection_time is not None and recovery_time is not None:
        recovery_index = 1 / (recovery_time - infection_time) if recovery_time > infection_time else 0
    
    # 获取代理人特质
    traits = user_traits[agent_name]
    
    # 创建数据条目
    agent_data = {
        'id': agent_id,
        'name': agent_name,
        'openness': traits['openness'],
        'conscientiousness': traits['conscientiousness'],
        'extraversion': traits['extraversion'],
        'agreeableness': traits['agreeableness'],
        'neuroticism': traits['neuroticism'],
        'education': traits['education'],
        'initial_belief': beliefs[0],
        'final_belief': beliefs[-1],
        'belief_changes': belief_changes,
        'infection_time': infection_time if infection_time is not None else -1,  # -1表示从未感染
        'recovery_time': recovery_time if recovery_time is not None else -1,     # -1表示从未恢复
        'total_infected_time': total_infected_time,
        'max_consecutive_infected': max_consecutive_infected,
        'resistance_index': resistance_index,
        'recovery_index': recovery_index
    }
    
    # 添加信念历史
    for i, belief in enumerate(beliefs):
        agent_data[f'belief_{i}'] = belief
    
    analysis_data.append(agent_data)

# 转换为DataFrame
df = pd.DataFrame(analysis_data)

# 显示基本统计信息
print(f"总代理人数: {len(df)}")
print(f"初始相信谣言者: {df[df['initial_belief'] == 1].shape[0]}")
print(f"初始不信谣言者: {df[df['initial_belief'] == 0].shape[0]}")
print(f"最终相信谣言者: {df[df['final_belief'] == 1].shape[0]}")
print(f"最终不信谣言者: {df[df['final_belief'] == 0].shape[0]}")
print(f"发生信念变化的代理人: {df[df['belief_changes'] > 0].shape[0]}")

# 保存处理后的数据
df.to_csv("analysis_results/goal1.3/personality_analysis_data.csv", index=False)

###########################################
# 第3步: 描述性统计分析
###########################################

# 特质分布
plt.figure(figsize=(15, 10))

# 大五人格特质的分布
personality_traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
for i, trait in enumerate(personality_traits):
    plt.subplot(2, 3, i+1)
    sns.countplot(x=trait, hue=trait, data=df, palette='viridis', legend=False)
    plt.title(f'{trait_names[trait]}分布')
    plt.xlabel('特质水平')
    plt.ylabel('数量')

plt.tight_layout()
plt.savefig("analysis_results/goal1.3/personality_traits_distribution.png", dpi=300)
plt.close()

# 信念变化指标的分布
plt.figure(figsize=(15, 10))
belief_metrics = ['belief_changes', 'infection_time', 'recovery_time', 
                  'total_infected_time', 'max_consecutive_infected']

for i, metric in enumerate(belief_metrics):
    plt.subplot(2, 3, i+1)
    
    # 剔除-1值（表示从未感染/恢复）
    if metric in ['infection_time', 'recovery_time']:
        valid_data = df[df[metric] >= 0]
    else:
        valid_data = df
    
    if not valid_data.empty:
        sns.histplot(valid_data[metric], kde=True, color='purple')
        plt.title(f'{metric}分布')
        plt.xlabel(metric)
        plt.ylabel('频率')

plt.tight_layout()
plt.savefig("analysis_results/goal1.3/belief_metrics_distribution.png", dpi=300)
plt.close()

###########################################
# 第4步: 相关性分析
###########################################

# 计算相关性矩阵
# 相关性分析变量：大五人格特质和信念指标
correlation_vars = personality_traits + ['belief_changes', 'infection_time', 'recovery_time', 
                                         'total_infected_time', 'max_consecutive_infected',
                                         'resistance_index', 'recovery_index']

# 移除infection_time和recovery_time中的-1值（表示从未感染/恢复）
corr_df = df.copy()
for col in ['infection_time', 'recovery_time']:
    corr_df = corr_df[corr_df[col] >= 0]

# 如果数据不足，使用原始数据
if len(corr_df) < 10:
    corr_df = df.copy()
    # 将-1替换为NaN，以便在计算相关性时忽略
    corr_df['infection_time'] = corr_df['infection_time'].replace(-1, np.nan)
    corr_df['recovery_time'] = corr_df['recovery_time'].replace(-1, np.nan)

# 计算相关性
corr_matrix = corr_df[correlation_vars].corr()

# 自定义相关性热图的颜色映射
cmap = LinearSegmentedColormap.from_list('custom_diverging', 
                                          ['#3498db', '#f1f1f1', '#e74c3c'], 
                                          N=256)

# 绘制相关性热图
plt.figure(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # 创建上三角掩码
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap=cmap, center=0,
            mask=mask, vmin=-1, vmax=1, linewidths=0.5)

# 使用中文特质名替换英文名
ytick_labels = [trait_names.get(label, label) for label in corr_matrix.index]
xtick_labels = [trait_names.get(label, label) for label in corr_matrix.columns]
plt.yticks(np.arange(0.5, len(corr_matrix.index) + 0.5), ytick_labels, rotation=0)
plt.xticks(np.arange(0.5, len(corr_matrix.columns) + 0.5), xtick_labels, rotation=45, ha='right')

plt.title('大五人格特质与信念指标的相关性矩阵', fontsize=16)
plt.tight_layout()
plt.savefig("analysis_results/goal1.3/correlation_matrix.png", dpi=300)
plt.close()

# 分析相关性显著性并输出结果
significant_correlations = []

for trait in personality_traits:
    for metric in ['belief_changes', 'infection_time', 'recovery_time', 
                   'total_infected_time', 'max_consecutive_infected',
                   'resistance_index', 'recovery_index']:
        # 移除缺失值
        valid_data = corr_df[[trait, metric]].dropna()
        
        if len(valid_data) >= 5:  # 确保有足够的数据点
            corr_coef, p_value = stats.pearsonr(valid_data[trait], valid_data[metric])
            
            significant_correlations.append({
                'trait': trait,
                'trait_cn': trait_names[trait],
                'metric': metric,
                'correlation': corr_coef,
                'p_value': p_value,
                'significant': p_value < 0.05
            })

# 转换为DataFrame并打印显著结果
sig_corr_df = pd.DataFrame(significant_correlations)
significant_results = sig_corr_df[sig_corr_df['significant']]

print("\n显著相关结果:")
print(significant_results[['trait_cn', 'metric', 'correlation', 'p_value']])

###########################################
# 第5步: 分组比较与方差分析
###########################################

# 对每个人格特质，比较不同特质水平的信念指标差异
for trait in personality_traits:
    plt.figure(figsize=(15, 12))
    
    # 信念指标
    belief_metrics = ['belief_changes', 'infection_time', 'recovery_time', 
                     'total_infected_time', 'max_consecutive_infected',
                     'resistance_index', 'recovery_index']
    
    for i, metric in enumerate(belief_metrics):
        plt.subplot(3, 3, i+1)
        
        # 准备数据：移除无效值
        if metric in ['infection_time', 'recovery_time']:
            valid_data = df[df[metric] >= 0].copy()
        else:
            valid_data = df.copy()
        
        if len(valid_data) >= 5:
            # 添加特质水平分组标签
            valid_data['trait_level'] = valid_data[trait].map({1: '低', 2: '中', 3: '高'})
            
            # 绘制箱线图
            sns.boxplot(x='trait_level', y=metric, hue='trait_level', data=valid_data, palette='viridis', legend=False)
            plt.title(f'{trait_names[trait]}水平对{metric}的影响')
            plt.xlabel(f'{trait_names[trait]}水平')
            plt.ylabel(metric)
            
            # 执行方差分析
            try:
                groups = []
                for level in [1, 2, 3]:
                    level_data = valid_data[valid_data[trait] == level][metric].dropna()
                    if len(level_data) > 0:
                        groups.append(level_data)
                
                if len(groups) >= 2 and all(len(g) > 1 for g in groups):
                    f_val, p_val = stats.f_oneway(*groups)
                    plt.annotate(f"ANOVA: F={f_val:.2f}, p={p_val:.4f}\n{'显著' if p_val < 0.05 else '不显著'}", 
                                xy=(0.05, 0.95), xycoords="axes fraction", 
                                bbox=dict(boxstyle="round", fc="white", alpha=0.8))
            except Exception as e:
                print(f"ANOVA分析出错: {e}")
                continue
    
    plt.tight_layout()
    plt.savefig(f"analysis_results/goal1.3/{trait_names[trait]}_group_analysis.png", dpi=300)
    plt.close()

###########################################
# 第6步: 回归分析
###########################################

# 对每个关键信念指标进行多元线性回归
regression_results = {}

for metric in ['belief_changes', 'total_infected_time', 'resistance_index', 'recovery_index']:
    # 准备数据
    X = df[personality_traits].copy()
    y = df[metric].copy()
    
    # 添加常数项
    X = sm.add_constant(X)
    
    # 拟合模型
    model = sm.OLS(y, X).fit()
    
    # 保存结果
    regression_results[metric] = {
        'model': model,
        'summary': model.summary(),
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj,
        'p_value': model.f_pvalue,
        'significant_traits': []
    }
    
    # 记录显著的特质
    for trait in personality_traits:
        if trait in model.pvalues and model.pvalues[trait] < 0.05:
            regression_results[metric]['significant_traits'].append({
                'trait': trait,
                'trait_cn': trait_names[trait],
                'coef': model.params[trait],
                'p_value': model.pvalues[trait]
            })
    
    # 保存回归结果
    with open(f"analysis_results/goal1.3/regression_{metric}.txt", "w", encoding="utf-8") as f:
        f.write(str(model.summary()))

# 创建回归结果汇总
regression_summary = []
for metric, result in regression_results.items():
    summary = {
        'metric': metric,
        'r_squared': result['r_squared'],
        'adj_r_squared': result['adj_r_squared'],
        'model_p_value': result['p_value'],
        'significant_traits': ', '.join([item['trait_cn'] for item in result['significant_traits']]) or '无'
    }
    regression_summary.append(summary)

# 转换为DataFrame
regression_df = pd.DataFrame(regression_summary)
print("\n回归分析汇总:")
print(regression_df)

# 绘制回归系数图
plt.figure(figsize=(15, 10))
for i, metric in enumerate(['belief_changes', 'total_infected_time', 'resistance_index', 'recovery_index']):
    plt.subplot(2, 2, i+1)
    
    result = regression_results[metric]
    model = result['model']
    
    # 提取大五人格特质的系数和置信区间
    coefs = [model.params[trait] for trait in personality_traits]
    conf_ints = model.conf_int()
    errors = [conf_ints[1][trait] - model.params[trait] for trait in personality_traits]
    
    # 绘制系数图
    colors = ['#3498db' if coef > 0 else '#e74c3c' for coef in coefs]
    
    plt.bar(range(len(personality_traits)), coefs, yerr=errors, color=colors,
            align='center', alpha=0.7, ecolor='black', capsize=10)
    
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.xticks(range(len(personality_traits)), [trait_names[trait] for trait in personality_traits], rotation=45)
    plt.title(f'{metric}的回归系数')
    plt.ylabel('系数值')
    
    # 添加R²和p值标注
    plt.annotate(f"R² = {model.rsquared:.3f}, p = {model.f_pvalue:.4f}", 
                xy=(0.05, 0.95), xycoords="axes fraction", 
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))

plt.tight_layout()
plt.savefig("analysis_results/goal1.3/regression_coefficients.png", dpi=300)
plt.close()

###########################################
# 第7步: 信念轨迹分析
###########################################

# 分析不同特质水平的信念变化轨迹
for trait in personality_traits:
    plt.figure(figsize=(10, 8))
    
    # 获取步数
    steps = [col for col in df.columns if col.startswith('belief_')]
    step_indices = list(range(len(steps)))
    print(f"时间步数量: {len(steps)}")  # 调试信息
    
    # 对每个特质水平，计算平均信念轨迹
    for level in [1, 2, 3]:
        level_df = df[df[trait] == level]
        if len(level_df) > 0:
            # 计算每个时间步的平均信念
            mean_beliefs = [level_df[steps[i]].mean() for i in range(len(steps))]
            
            # 绘制平均轨迹
            plt.plot(step_indices, mean_beliefs, 
                    marker='o', linestyle='-', linewidth=2,
                    label=f'{trait_names[trait]}={level} (N={len(level_df)})')
    
    plt.xlabel('时间步')
    plt.ylabel('平均信念 (0=不信, 1=相信)')
    plt.title(f'不同{trait_names[trait]}水平的信念变化轨迹')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(step_indices)
    plt.yticks([0, 0.25, 0.5, 0.75, 1])
    
    plt.tight_layout()
    plt.savefig(f"analysis_results/goal1.3/{trait_names[trait]}_belief_trajectory.png", dpi=300)
    plt.close()

###########################################
# 第8步: 特质交互效应分析
###########################################

# 探索两个特质的交互效应
for i, trait1 in enumerate(personality_traits):
    for trait2 in personality_traits[i+1:]:
        plt.figure(figsize=(12, 10))
        
        # 为每个信念指标创建子图
        for i, metric in enumerate(['belief_changes', 'total_infected_time', 'resistance_index']):
            plt.subplot(2, 2, i+1)
            
            # 创建交互特征
            interaction_df = df.copy()
            interaction_df['trait_combo'] = interaction_df.apply(
                lambda row: f"{trait_names[trait1]}={row[trait1]}, {trait_names[trait2]}={row[trait2]}", 
                axis=1
            )
            
            # 计算每个组合的平均值
            combo_means = interaction_df.groupby('trait_combo')[metric].mean().reset_index()
            
            # 绘制条形图
            sns.barplot(x='trait_combo', y=metric, hue='trait_combo', data=combo_means, palette='viridis', legend=False)
            plt.title(f'{trait_names[trait1]}与{trait_names[trait2]}对{metric}的交互效应')
            plt.xticks(rotation=90)
            plt.tight_layout()
        
        plt.savefig(f"analysis_results/goal1.3/{trait_names[trait1]}_{trait_names[trait2]}_interaction.png", dpi=300)
        plt.close()

###########################################
# 第9步: 关键发现总结
###########################################

# 找出对各指标影响最大的特质
influential_traits = {}

# 基于相关系数评估影响
for trait in personality_traits:
    abs_correlations = []
    for row in significant_correlations:
        if row['trait'] == trait and row['significant']:
            abs_correlations.append(abs(row['correlation']))
    
    # 计算平均绝对相关系数
    if abs_correlations:
        influential_traits[trait] = sum(abs_correlations) / len(abs_correlations)
    else:
        influential_traits[trait] = 0

# 排序找出影响最大的特质
sorted_influential = sorted(influential_traits.items(), key=lambda x: x[1], reverse=True)
most_influential = sorted_influential[0][0] if sorted_influential else None

# 找出与各关键指标显著相关的特质
significant_traits = {}
for metric in ['belief_changes', 'total_infected_time', 'resistance_index', 'recovery_index']:
    significant_for_metric = []
    for row in significant_correlations:
        if row['metric'] == metric and row['significant']:
            significant_for_metric.append({
                'trait': row['trait'],
                'trait_cn': row['trait_cn'],
                'correlation': row['correlation']
            })
    
    significant_traits[metric] = significant_for_metric

# 创建研究发现摘要
findings_summary = "# 大五人格特质对虚假健康信息感染与恢复过程的影响分析\n\n"

# 基本统计信息
findings_summary += "## 1. 基本统计信息\n\n"
findings_summary += f"- 总代理人数: {len(df)}\n"
findings_summary += f"- 初始相信谣言者: {df[df['initial_belief'] == 1].shape[0]}\n"
findings_summary += f"- 初始不信谣言者: {df[df['initial_belief'] == 0].shape[0]}\n"
findings_summary += f"- 最终相信谣言者: {df[df['final_belief'] == 1].shape[0]}\n"
findings_summary += f"- 最终不信谣言者: {df[df['final_belief'] == 0].shape[0]}\n"
findings_summary += f"- 发生信念变化的代理人: {df[df['belief_changes'] > 0].shape[0]}\n\n"

# 显著相关结果
findings_summary += "## 2. 显著相关结果\n\n"
if not significant_results.empty:
    findings_summary += "| 特质 | 指标 | 相关系数 | p值 |\n"
    findings_summary += "|-----|-----|---------|----|\n"
    for _, row in significant_results.iterrows():
        findings_summary += f"| {row['trait_cn']} | {row['metric']} | {row['correlation']:.3f} | {row['p_value']:.4f} |\n"
else:
    findings_summary += "未发现显著相关关系。\n"

# 回归分析结果
findings_summary += "\n## 3. 回归分析结果\n\n"
findings_summary += "| 指标 | R² | 调整R² | 模型p值 | 显著特质 |\n"
findings_summary += "|-----|---|-------|-------|--------|\n"
for _, row in regression_df.iterrows():
    findings_summary += f"| {row['metric']} | {row['r_squared']:.3f} | {row['adj_r_squared']:.3f} | {row['model_p_value']:.4f} | {row['significant_traits']} |\n"

# 关键发现
findings_summary += "\n## 4. 关键发现\n\n"

if most_influential:
    findings_summary += f"### 4.1 最具影响力的人格特质\n\n"
    findings_summary += f"- **{trait_names[most_influential]}** 对谣言传播过程影响最大，平均绝对相关系数为 {influential_traits[most_influential]:.3f}\n\n"

findings_summary += "### 4.2 各指标的关键影响因素\n\n"

for metric, traits in significant_traits.items():
    if traits:
        findings_summary += f"#### {metric}:\n"
        for trait_info in traits:
            direction = "正相关" if trait_info['correlation'] > 0 else "负相关"
            findings_summary += f"- **{trait_info['trait_cn']}** 与 {metric} 呈 {direction} (r={trait_info['correlation']:.3f})\n"
        findings_summary += "\n"

# 结论和建议
findings_summary += "## 5. 结论与建议\n\n"

# 基于发现生成结论
if significant_results.empty:
    findings_summary += "本研究没有发现大五人格特质与谣言传播指标之间存在显著相关关系，可能需要更大样本量或更精细的指标设计。\n\n"
else:
    if most_influential:
        trait_cn = trait_names[most_influential]
        # 根据最具影响力特质对应的相关方向，提供不同建议
        positive_influence = False
        for row in significant_correlations:
            if row['trait'] == most_influential and row['significant'] and row['correlation'] > 0:
                positive_influence = True
                break
        
        if positive_influence:
            findings_summary += f"1. **{trait_cn}** 特质较高的个体可能更容易受到虚假健康信息的影响，健康教育时应特别关注此类人群。\n\n"
        else:
            findings_summary += f"1. **{trait_cn}** 特质较高的个体可能对虚假健康信息具有更强的抵抗力，可以借鉴其信息处理方式。\n\n"
    
    # 基于回归分析结果提供建议
    best_model = max(regression_results.items(), key=lambda x: x[1]['r_squared'])
    if best_model[1]['r_squared'] > 0.1:  # R²至少0.1才有一定解释力
        findings_summary += f"2. 大五人格特质能解释{best_model[1]['r_squared'] * 100:.1f}%的{best_model[0]}变异，说明个性特质在虚假健康信息传播中起着重要作用。\n\n"
    
    # 防范谣言传播的建议
    findings_summary += "3. 防范虚假健康信息传播的策略建议：\n"
    
    # 根据具体发现提供个性化建议
    resistance_traits = []
    vulnerability_traits = []
    
    for row in significant_correlations:
        if row['significant']:
            if (row['metric'] == 'resistance_index' and row['correlation'] > 0) or \
               (row['metric'] in ['infection_time', 'recovery_index'] and row['correlation'] > 0) or \
               (row['metric'] in ['belief_changes', 'total_infected_time'] and row['correlation'] < 0):
                resistance_traits.append(row['trait_cn'])
            
            if (row['metric'] == 'resistance_index' and row['correlation'] < 0) or \
               (row['metric'] in ['infection_time', 'recovery_index'] and row['correlation'] < 0) or \
               (row['metric'] in ['belief_changes', 'total_infected_time'] and row['correlation'] > 0):
                vulnerability_traits.append(row['trait_cn'])
    
    # 移除重复项
    resistance_traits = list(set(resistance_traits))
    vulnerability_traits = list(set(vulnerability_traits))
    
    if resistance_traits:
        findings_summary += f"   - 针对{', '.join(resistance_traits)}特质较高的人群：可作为健康信息的意见领袖，帮助传播科学健康观念\n"
    
    if vulnerability_traits:
        findings_summary += f"   - 针对{', '.join(vulnerability_traits)}特质较高的人群：提供更多科学证据和批判性思维训练，增强对虚假信息的识别能力\n"
    
    findings_summary += "   - 设计针对不同人格特质的差异化健康信息传播策略，提高信息接受度\n"
    findings_summary += "   - 在健康信息传播中考虑人格特质因素，对高风险人群进行重点干预\n\n"

# 保存研究发现摘要
with open("analysis_results/goal1.3/findings_summary.md", "w", encoding="utf-8") as f:
    f.write(findings_summary)

###########################################
# 第10步: 信念轨迹可视化 - 热图
###########################################

# 创建每个代理人的信念轨迹热图
plt.figure(figsize=(12, 8))

# 提取所有步骤的信念数据
belief_cols = [col for col in df.columns if col.startswith('belief_')]
belief_matrix = df[belief_cols].values

# 按特质分组排序
for trait in personality_traits:
    # 按特质值对代理人排序
    sorted_indices = df[trait].argsort()
    sorted_matrix = belief_matrix[sorted_indices]
    sorted_trait_values = df[trait].iloc[sorted_indices].values
    
    plt.figure(figsize=(12, 8))
    
    # 创建热图
    cmap = sns.color_palette(["#3498db", "#e74c3c"], as_cmap=True)
    ax = sns.heatmap(sorted_matrix, cmap=cmap, cbar_kws={'label': '信念状态 (0=不信, 1=相信)'})
    
    # 添加特质水平标记
    trait_levels = []
    current_level = sorted_trait_values[0]
    level_positions = [0]
    
    for i, level in enumerate(sorted_trait_values[1:], 1):
        if level != current_level:
            trait_levels.append(current_level)
            current_level = level
            level_positions.append(i)
    
    trait_levels.append(current_level)
    level_positions.append(len(sorted_trait_values))
    
    # 在热图旁边添加特质水平标记
    for i in range(len(trait_levels)):
        level = trait_levels[i]
        pos = (level_positions[i] + level_positions[i+1]) / 2 if i < len(trait_levels) - 1 else level_positions[i]
        plt.text(-0.5, pos, f'{trait_names[trait]}={level}', 
                 verticalalignment='center', horizontalalignment='right')
    
    plt.xlabel('时间步')
    plt.ylabel('代理人 (按特质水平排序)')
    plt.title(f'按{trait_names[trait]}水平排序的信念轨迹热图')
    
    # 重新设置y轴刻度，使其更清晰
    plt.yticks([])
    
    plt.tight_layout()
    plt.savefig(f"analysis_results/goal1.3/{trait_names[trait]}_belief_heatmap.png", dpi=300)
    plt.close()

###########################################
# 第11步: 特质组合分析
###########################################

# 分析两个关键特质的组合效应
# 选择两个影响最大的特质
if len(sorted_influential) >= 2:
    top_traits = [t[0] for t in sorted_influential[:2]]
    trait1, trait2 = top_traits
    
    plt.figure(figsize=(15, 12))
    
    # 对几个关键指标进行分析
    for i, metric in enumerate(['belief_changes', 'total_infected_time', 'resistance_index', 'recovery_index']):
        plt.subplot(2, 2, i+1)
        
        # 创建交叉表
        combo_df = df.pivot_table(
            values=metric, 
            index=trait1, 
            columns=trait2, 
            aggfunc='mean'
        )
        
        # 热图显示组合效应
        sns.heatmap(combo_df, annot=True, fmt=".2f", cmap="viridis")
        plt.xlabel(f'{trait_names[trait2]}水平')
        plt.ylabel(f'{trait_names[trait1]}水平')
        plt.title(f'{trait_names[trait1]}与{trait_names[trait2]}对{metric}的联合影响')
    
    plt.tight_layout()
    plt.savefig(f"analysis_results/goal1.3/top_traits_combination_analysis.png", dpi=300)
    plt.close()

###########################################
# 第12步: 代理人信念变化行为分析
###########################################

# 分析不同行为类型的代理人特质差异
behavior_types = [
    {'name': '稳定不信者', 'condition': (df['belief_changes'] == 0) & (df['initial_belief'] == 0)},
    {'name': '稳定相信者', 'condition': (df['belief_changes'] == 0) & (df['initial_belief'] == 1)},
    {'name': '易感染者', 'condition': (df['belief_changes'] > 0) & (df['infection_time'] >= 0) & (df['infection_time'] <= 5)},
    {'name': '迟感染者', 'condition': (df['belief_changes'] > 0) & (df['infection_time'] > 5)},
    {'name': '快恢复者', 'condition': (df['belief_changes'] > 0) & (df['recovery_time'] >= 0) & (df['recovery_time'] - df['infection_time'] <= 3)},
    {'name': '迟恢复者', 'condition': (df['belief_changes'] > 0) & (df['recovery_time'] >= 0) & (df['recovery_time'] - df['infection_time'] > 3)}
]

# 计算每种行为类型的平均特质值
behavior_trait_means = []

for behavior in behavior_types:
    subset = df[behavior['condition']]
    if len(subset) > 0:
        means = {trait: subset[trait].mean() for trait in personality_traits}
        means['behavior_type'] = behavior['name']
        means['count'] = len(subset)
        behavior_trait_means.append(means)

# 转换为DataFrame
behavior_df = pd.DataFrame(behavior_trait_means)

if not behavior_df.empty:
    # 可视化各行为类型的特质分布
    plt.figure(figsize=(15, 10))
    
    # 绘制分组柱状图
    x = np.arange(len(behavior_df))
    width = 0.15
    offsets = [-2, -1, 0, 1, 2]
    
    for i, trait in enumerate(personality_traits):
        plt.bar(x + offsets[i] * width, behavior_df[trait], width, label=trait_names[trait])
    
    plt.xlabel('行为类型')
    plt.ylabel('平均特质水平')
    plt.title('不同信念变化行为类型的平均人格特质水平')
    plt.xticks(x, behavior_df['behavior_type'])
    plt.legend()
    
    # 添加样本量标注
    for i, count in enumerate(behavior_df['count']):
        plt.text(i, 0.2, f"n={count}", ha='center')
    
    plt.tight_layout()
    plt.savefig("analysis_results/goal1.3/behavior_types_traits.png", dpi=300)
    plt.close()

    # 保存行为类型数据
    behavior_df.to_csv("analysis_results/goal1.3/behavior_types_analysis.csv", index=False)

print("\n分析完成! 结果已保存到 analysis_results/goal1.3/ 目录")
print("主要输出文件:")
print("1. personality_analysis_data.csv - 处理后的分析数据")
print("2. correlation_matrix.png - 相关性热图")
print("3. regression_coefficients.png - 回归系数图")
print("4. *_belief_trajectory.png - 信念轨迹图")
print("5. *_belief_heatmap.png - 信念热图")
print("6. findings_summary.md - 研究发现摘要")

# 添加在分析末尾
print("\n===== 扩展分析: 探索数据中的模式 =====")

# 1. 检查样本大小和分布
print("\n特质水平分布:")
for trait in personality_traits:
    trait_counts = df[trait].value_counts().sort_index()
    print(f"{trait_names[trait]}: {trait_counts.to_dict()}")

# 2. 检查信念变化模式
print("\n信念变化模式:")
# 计算初始不信且变为相信的比例
initially_susceptible = df[df['initial_belief'] == 0]
became_infected = initially_susceptible[initially_susceptible['total_infected_time'] > 0]
print(f"初始不信且后来相信的比例: {len(became_infected)/len(initially_susceptible)*100:.1f}%")

# 计算初始相信且变为不信的比例
initially_infected = df[df['initial_belief'] == 1]
became_recovered = initially_infected[initially_infected['final_belief'] == 0]
print(f"初始相信且后来不信的比例: {len(became_recovered)/len(initially_infected)*100:.1f}%")

# 3. 效应量分析（即使不显著）
print("\n效应量分析 (相关系数大小):")
effect_sizes = []
for trait in personality_traits:
    for metric in ['belief_changes', 'total_infected_time', 'resistance_index']:
        valid_data = df[[trait, metric]].dropna()
        if len(valid_data) >= 5:
            corr, _ = stats.pearsonr(valid_data[trait], valid_data[metric])
            effect_sizes.append({
                'trait': trait,
                'trait_cn': trait_names[trait],
                'metric': metric,
                'correlation': corr,
                'abs_corr': abs(corr)
            })

effect_df = pd.DataFrame(effect_sizes)
if not effect_df.empty:
    # 找出最强的关系，即使不显著
    strongest = effect_df.sort_values('abs_corr', ascending=False).head(5)
    print("最强的关系 (即使不显著):")
    for _, row in strongest.iterrows():
        direction = "正" if row['correlation'] > 0 else "负"
        print(f"  {row['trait_cn']} 与 {row['metric']}: {direction}相关 (r={row['correlation']:.3f})")

# 4. 描述性统计：按特质水平分组
print("\n各特质水平的信念变化指标均值:")
for trait in personality_traits:
    print(f"\n{trait_names[trait]}:")
    for level in sorted(df[trait].unique()):
        level_df = df[df[trait] == level]
        if len(level_df) > 0:
            print(f"  水平={level} (n={len(level_df)}):")
            for metric in ['belief_changes', 'total_infected_time', 'resistance_index']:
                print(f"    {metric}: {level_df[metric].mean():.3f}")

# 添加在分析结束前
print("\n===== 替代分析: 非线性和分类方法 =====")

# 1. 将连续变量转换为分类变量
df['belief_change_cat'] = pd.cut(df['belief_changes'], 
                                [0, 1, 2, float('inf')], 
                                labels=['无变化', '变化一次', '多次变化'])

# 2. 卡方检验：检查特质水平与信念变化类别的关联
for trait in personality_traits:
    # 创建列联表
    contingency = pd.crosstab(df[trait], df['belief_change_cat'])
    if contingency.shape[0] > 1 and contingency.shape[1] > 1:
        # 执行卡方检验
        chi2, p, dof, expected = chi2_contingency(contingency)
        print(f"\n{trait_names[trait]}与信念变化类别的关联:")
        print(f"  卡方值={chi2:.3f}, p值={p:.3f}")
        print("  列联表:")
        print(contingency)

# 3. 简单决策树（分类方法）
from sklearn.tree import DecisionTreeClassifier, export_text
# 预测是否会发生信念变化
df['had_change'] = (df['belief_changes'] > 0).astype(int)
X = df[personality_traits]
y = df['had_change']

# 训练简单决策树
tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_model.fit(X, y)

# 打印决策规则
print("\n决策树规则 (预测是否会发生信念变化):")
tree_rules = export_text(tree_model, feature_names=personality_traits)
print(tree_rules)

# 4. 特征重要性
importances = tree_model.feature_importances_
indices = np.argsort(importances)[::-1]
print("\n特征重要性 (决策树):")
for i in indices:
    if importances[i] > 0:
        print(f"  {trait_names[personality_traits[i]]}: {importances[i]:.3f}")

# 绘制特质均值比较
plt.figure(figsize=(12, 8))
# 将代理人分为有变化和无变化两组
df['change_group'] = df['belief_changes'].apply(lambda x: '发生变化' if x > 0 else '无变化')

# 比较两组的特质均值
trait_means = df.groupby('change_group')[personality_traits].mean().reset_index()
trait_means_melted = pd.melt(trait_means, id_vars='change_group', value_vars=personality_traits,
                            var_name='trait', value_name='mean_value')
trait_means_melted['trait_cn'] = trait_means_melted['trait'].map(trait_names)

# 绘制分组柱状图
sns.barplot(x='trait_cn', y='mean_value', hue='change_group', data=trait_means_melted)
plt.title('信念变化组与无变化组的平均特质比较')
plt.xlabel('人格特质')
plt.ylabel('平均值')
plt.ylim(1, 3)
plt.legend(title='')
plt.savefig("analysis_results/goal1.3/trait_means_by_change_group.png", dpi=300)
plt.close()