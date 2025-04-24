import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import seaborn as sns
import os

# 为macOS尝试多个可能的中文字体
possible_fonts = [
    '/System/Library/Fonts/PingFang.ttc',
    '/Library/Fonts/Arial Unicode.ttf',
    '/System/Library/Fonts/STHeiti Light.ttc',
    '/System/Library/Fonts/STHeiti Medium.ttc',
    '/System/Library/Fonts/Hiragino Sans GB.ttc',
    '/System/Library/Fonts/AppleGothic.ttf',
    '/Library/Fonts/Microsoft/SimHei.ttf'  # 如果安装了Microsoft Office
]

chinese_font = None
for font_path in possible_fonts:
    if os.path.exists(font_path):
        chinese_font = fm.FontProperties(fname=font_path)
        print(f"使用中文字体: {font_path}")
        break

if chinese_font is None:
    # 尝试使用系统字体名称
    for font_name in ['PingFang SC', 'Heiti SC', 'STHeiti', 'SimHei', 'Microsoft YaHei']:
        try:
            fm.findfont(fm.FontProperties(family=font_name))
            chinese_font = fm.FontProperties(family=font_name)
            print(f"使用系统字体: {font_name}")
            break
        except:
            continue

if chinese_font is None:
    print("无法找到中文字体，将使用英文")
    exit()

# 读取三组实验数据
df1 = pd.read_csv("analysis_results/goal3.1/propagation_curves.csv")
df2 = pd.read_csv("analysis_results/goal3.2/propagation_curves.csv")
df3 = pd.read_csv("analysis_results/goal3.3/propagation_curves.csv")

# 读取参数数据
params1 = pd.read_csv("analysis_results/goal3.1/sir_parameters.csv")
params2 = pd.read_csv("analysis_results/goal3.2/sir_parameters.csv")
params3 = pd.read_csv("analysis_results/goal3.3/sir_parameters.csv")

# 获取R0值
r0_1 = params1[params1['Parameter'] == 'R0']['Value'].values[0]
r0_2 = params2[params2['Parameter'] == 'R0']['Value'].values[0]
r0_3 = params3[params3['Parameter'] == 'R0']['Value'].values[0]

# 创建图表
plt.figure(figsize=(20, 18))  # 增加高度
sns.set_style("whitegrid")

# 创建三个子图
ax1 = plt.subplot(3, 1, 1)
ax1.set_position([0.1, 0.68, 0.8, 0.28])  # 增加每个子图的高度
ax2 = plt.subplot(3, 1, 2)
ax2.set_position([0.1, 0.38, 0.8, 0.28])
ax3 = plt.subplot(3, 1, 3)
ax3.set_position([0.1, 0.08, 0.8, 0.28])

# 绘制第一组实验
ax1.plot(df1['Time_Step'], df1['LLM_S'], 'b-', linewidth=2.5, label='LLM: S')
ax1.plot(df1['Time_Step'], df1['LLM_I'], 'r-', linewidth=2.5, label='LLM: I')
ax1.plot(df1['Time_Step'], df1['LLM_R'], 'g-', linewidth=2.5, label='LLM: R')
ax1.plot(df1['Time_Step'], df1['SIR_S'], 'b--', linewidth=1.8, label='SIR: S')
ax1.plot(df1['Time_Step'], df1['SIR_I'], 'r--', linewidth=1.8, label='SIR: I')
ax1.plot(df1['Time_Step'], df1['SIR_R'], 'g--', linewidth=1.8, label='SIR: R')
ax1.set_title(f'实验组1: R₀ = {r0_1:.2f}', fontsize=0, fontproperties=chinese_font)
ax1.set_ylabel('人口比例', fontsize=20, fontproperties=chinese_font)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.text(0.02, 0.95, '峰值(SIR): 0.80', transform=ax1.transAxes, fontsize=15, 
         bbox=dict(facecolor='white', alpha=0.7), fontproperties=chinese_font)
ax1.text(0.02, 0.88, '峰值(LLM): 0.20', transform=ax1.transAxes, fontsize=15, 
         bbox=dict(facecolor='white', alpha=0.7), fontproperties=chinese_font)
ax1.title.set_fontsize(30)

# 绘制第二组实验
ax2.plot(df2['Time_Step'], df2['LLM_S'], 'b-', linewidth=2.5)
ax2.plot(df2['Time_Step'], df2['LLM_I'], 'r-', linewidth=2.5)
ax2.plot(df2['Time_Step'], df2['LLM_R'], 'g-', linewidth=2.5)
ax2.plot(df2['Time_Step'], df2['SIR_S'], 'b--', linewidth=1.8)
ax2.plot(df2['Time_Step'], df2['SIR_I'], 'r--', linewidth=1.8)
ax2.plot(df2['Time_Step'], df2['SIR_R'], 'g--', linewidth=1.8)
ax2.set_title(f'实验组2: R₀ = {r0_2:.2f}', fontsize=30, fontproperties=chinese_font)
ax2.set_ylabel('人口比例', fontsize=20, fontproperties=chinese_font)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.text(0.02, 0.95, '峰值(SIR): 0.86', transform=ax2.transAxes, fontsize=15, 
         bbox=dict(facecolor='white', alpha=0.7), fontproperties=chinese_font)
ax2.text(0.02, 0.88, '峰值(LLM): 0.35', transform=ax2.transAxes, fontsize=15, 
         bbox=dict(facecolor='white', alpha=0.7), fontproperties=chinese_font)
ax2.title.set_fontsize(30)

# 绘制第三组实验
ax3.plot(df3['Time_Step'], df3['LLM_S'], 'b-', linewidth=2.5)
ax3.plot(df3['Time_Step'], df3['LLM_I'], 'r-', linewidth=2.5)
ax3.plot(df3['Time_Step'], df3['LLM_R'], 'g-', linewidth=2.5)
ax3.plot(df3['Time_Step'], df3['SIR_S'], 'b--', linewidth=1.8)
ax3.plot(df3['Time_Step'], df3['SIR_I'], 'r--', linewidth=1.8)
ax3.plot(df3['Time_Step'], df3['SIR_R'], 'g--', linewidth=1.8)
ax3.set_title(f'实验组3: R₀ = {r0_3:.2f}', fontsize=30, fontproperties=chinese_font)
ax3.set_xlabel('时间步', fontsize=20, fontproperties=chinese_font)
ax3.set_ylabel('人口比例', fontsize=20, fontproperties=chinese_font)
ax3.grid(True, linestyle='--', alpha=0.7)
ax3.text(0.02, 0.95, '峰值(SIR): 0.84', transform=ax3.transAxes, fontsize=15, 
         bbox=dict(facecolor='white', alpha=0.7), fontproperties=chinese_font)
ax3.text(0.02, 0.88, '峰值(LLM): 0.40', transform=ax3.transAxes, fontsize=15, 
         bbox=dict(facecolor='white', alpha=0.7), fontproperties=chinese_font)
ax3.title.set_fontsize(30)

# 添加峰值时间标记
ax1.axvline(x=2, color='r', linestyle=':', alpha=0.5)
ax2.axvline(x=2, color='r', linestyle=':', alpha=0.5)
ax2.axvline(x=5, color='r', linestyle=':', alpha=0.5)
ax3.axvline(x=2, color='r', linestyle=':', alpha=0.5)

# 添加共享图例
handles, labels = ax1.get_legend_handles_labels()
fig = plt.gcf()

# 将图例移至图表底部，减小空白区域
leg = fig.legend(handles, labels, loc='upper center', 
                bbox_to_anchor=(0.5, 0.08),  # 将y坐标从0.05改为0.02
                fancybox=True, shadow=True, ncol=6)
for text in leg.get_texts():
    text.set_fontproperties(chinese_font)
    text.set_fontsize(30)

# 调整子图之间的间距，缩小底部空白
plt.tight_layout(rect=[0, 0.04, 1, 0.97])  # 将底部边界从0.07改为0.04

# 注释放置位置调整
fig.text(0.5, 0.005,  # 将y坐标从0.01改为0.005
        '注: 实线表示LLM模型，虚线表示SIR模型；蓝色=易感人群(S)，红色=感染者(I)，绿色=恢复者(R)',
        ha='center', fontsize=30, fontproperties=chinese_font)

# 保存图片
plt.savefig('llm_sir_comparison_combined.png', dpi=600, bbox_inches='tight')

# 显示图片
# plt.show()
plt.clf()
plt.close('all')