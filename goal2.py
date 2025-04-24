import pandas as pd
import statsmodels.formula.api as smf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.font_manager as fm

# 创建结果目录
os.makedirs('analysis_results/goal2.3', exist_ok=True)

# 使用已有的数据
df = pd.read_csv('analysis_results/goal1.3/personality_analysis_data.csv')

# 检查数据
print(f"数据集形状: {df.shape}")
print("\n教育背景分布:")
print(df['education'].value_counts().sort_index())

# 检查是否有缺失值
print("\n各列缺失值数量:")
print(df.isnull().sum())

# 创建交互项
df['openness_x_education'] = df['openness'] * df['education']
df['conscientiousness_x_education'] = df['conscientiousness'] * df['education']
df['extraversion_x_education'] = df['extraversion'] * df['education']
df['agreeableness_x_education'] = df['agreeableness'] * df['education']
df['neuroticism_x_education'] = df['neuroticism'] * df['education']

# 定义要分析的结果变量
outcome_variables = ['belief_changes', 'total_infected_time', 'resistance_index', 'recovery_index']

# 创建数据框来存储结果
results_df = pd.DataFrame(columns=['outcome_variable', 'trait', 'interaction_p_value', 'interaction_coef', 'significant'])

# 在绘图代码前添加以下内容
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei']  # 优先尝试这些字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 对每个结果变量进行分析
for outcome in outcome_variables:
    print(f"\n\n==== 分析 {outcome} ====")
    
    # 全模型（包含所有特质和交互项）
    formula = f"""
    {outcome} ~ openness + conscientiousness + extraversion + agreeableness + neuroticism + education +
    openness_x_education + conscientiousness_x_education + extraversion_x_education +
    agreeableness_x_education + neuroticism_x_education
    """
    model = smf.ols(formula, data=df).fit()
    
    print("\n全模型摘要:")
    print(f"R-squared: {model.rsquared:.4f}, Adjusted R-squared: {model.rsquared_adj:.4f}")
    print(f"F-statistic: {model.fvalue:.4f}, p-value: {model.f_pvalue:.4f}")
    
    # 检查交互项的显著性
    interaction_terms = ['openness_x_education', 'conscientiousness_x_education', 
                         'extraversion_x_education', 'agreeableness_x_education', 
                         'neuroticism_x_education']
    
    print("\n交互项系数和显著性:")
    for term in interaction_terms:
        if term in model.params.index:
            p_value = model.pvalues[term]
            coef = model.params[term]
            significant = p_value < 0.05
            trait = term.split('_x_')[0]
            
            print(f"{term}: 系数 = {coef:.4f}, p值 = {p_value:.4f}, 显著性: {'是' if significant else '否'}")
            
            # 将结果添加到数据框
            results_df = results_df._append({
                'outcome_variable': outcome,
                'trait': trait,
                'interaction_p_value': p_value,
                'interaction_coef': coef,
                'significant': significant
            }, ignore_index=True)
            
            # 如果交互项显著，绘制简单斜率图
            if significant:
                trait_name = term.split('_x_')[0]
                trait_mean = df[trait_name].mean()
                trait_std = df[trait_name].std()
                education_mean = df['education'].mean()
                education_std = df['education'].std()
                
                # 计算三个教育水平点
                low_edu = max(1, round(education_mean - education_std))
                med_edu = round(education_mean)
                high_edu = min(5, round(education_mean + education_std))
                
                # 创建特质值范围
                trait_range = np.linspace(df[trait_name].min(), df[trait_name].max(), 100)
                
                # 获取系数
                trait_coef = model.params[trait_name]
                edu_coef = model.params['education']
                interaction_coef = model.params[term]
                
                # 确保截距存在于模型中
                if 'Intercept' in model.params.index:
                    intercept = model.params['Intercept']
                else:
                    intercept = 0
                
                # 计算预测值
                y_low = intercept + trait_coef * trait_range + edu_coef * low_edu + interaction_coef * trait_range * low_edu
                y_med = intercept + trait_coef * trait_range + edu_coef * med_edu + interaction_coef * trait_range * med_edu
                y_high = intercept + trait_coef * trait_range + edu_coef * high_edu + interaction_coef * trait_range * high_edu
                
                # 绘制简单斜率图
                plt.figure(figsize=(10, 6))
                plt.plot(trait_range, y_low, 'b-', label=f'低教育背景 (教育水平 = {low_edu})')
                plt.plot(trait_range, y_med, 'g-', label=f'中等教育背景 (教育水平 = {med_edu})')
                plt.plot(trait_range, y_high, 'r-', label=f'高教育背景 (教育水平 = {high_edu})')
                
                trait_names = {
                    'openness': '开放性',
                    'conscientiousness': '尽责性',
                    'extraversion': '外向性',
                    'agreeableness': '宜人性',
                    'neuroticism': '神经质'
                }
                
                outcome_names = {
                    'belief_changes': '信念变化次数',
                    'total_infected_time': '总感染时间',
                    'resistance_index': '抵抗指数',
                    'recovery_index': '恢复指数'
                }
                
                plt.xlabel(trait_names.get(trait_name, trait_name))
                plt.ylabel(outcome_names.get(outcome, outcome))
                plt.title(f'{trait_names.get(trait_name, trait_name)}与教育背景对{outcome_names.get(outcome, outcome)}的交互作用')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(f'analysis_results/goal2.3/{trait_name}_{outcome}_interaction.png')
                plt.close()
    
    # 单独的调节分析
    print("\n单独的调节分析结果：")
    for trait in ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']:
        interaction_term = f"{trait}_x_education"
        formula_single = f"{outcome} ~ {trait} + education + {interaction_term}"
        model_single = smf.ols(formula_single, data=df).fit()
        
        p_value = model_single.pvalues[interaction_term] if interaction_term in model_single.pvalues.index else np.nan
        coef = model_single.params[interaction_term] if interaction_term in model_single.params.index else np.nan
        
        print(f"\n{trait}与教育背景的单独调节分析:")
        print(f"R-squared: {model_single.rsquared:.4f}")
        print(f"交互项p值: {p_value:.4f}")
        print(f"交互项系数: {coef:.4f}")

    # 在交互项分析之后添加
    print("\n主效应系数和显著性:")
    main_effects = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism', 'education']
    for term in main_effects:
        if term in model.params.index:
            p_value = model.pvalues[term]
            coef = model.params[term]
            significant = p_value < 0.05
            print(f"{term}: 系数 = {coef:.4f}, p值 = {p_value:.4f}, 显著性: {'是' if significant else '否'}")

# 创建一个热图显示调节效应
plt.figure(figsize=(12, 8))
heatmap_data = pd.pivot_table(
    results_df, 
    values='interaction_p_value', 
    index='trait',
    columns='outcome_variable'
)
sns.heatmap(heatmap_data, annot=True, cmap='coolwarm_r', fmt='.4f', vmin=0, vmax=0.1)
plt.title('人格特质与教育背景交互作用的显著性 (p值)')
plt.tight_layout()
plt.savefig('analysis_results/goal2.3/interaction_significance_heatmap.png')
plt.close()

# 保存结果
results_df.to_csv('analysis_results/goal2.3/moderation_results.csv', index=False)
print("\n分析完成。结果已保存到 analysis_results/goal2.3/ 目录")