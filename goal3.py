import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy.optimize import curve_fit
import seaborn as sns
import matplotlib
import platform
import os
import matplotlib as mpl
import matplotlib.font_manager as fm

# 创建结果保存目录
os.makedirs("analysis_results/goal3.3", exist_ok=True)

# 使用与goal1相同的字体设置方法
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
    chinese_font_available = True
except:
    print("无法找到中文字体，使用默认设置")
    chinese_font_available = False

# 如果找不到中文字体，使用英文标题（避免乱码）
def safe_title(zh_title, en_title):
    """根据中文字体可用性返回适当的标题"""
    return zh_title if chinese_font_available else en_title

# 加载LLM模拟的结果数据
def load_llm_results(file_path):
    """加载LLM模拟结果数据"""
    df = pd.read_csv(file_path)
    
    # 提取时间序列数据
    steps = range(16)  # 假设有16个时间步
    S_data = []
    I_data = []
    R_data = []
    
    # 统计每一步的S、I、R人数
    for step in steps:
        belief_col = f'belief_{step}'
        
        if belief_col in df.columns:
            # 只统计当前步骤中的信念值
            susceptible = sum((df[belief_col] == 0) & (df['initial_belief'] == 0))
            infected = sum(df[belief_col] == 1)
            recovered = sum((df[belief_col] == 0) & (df['initial_belief'] == 1))
            
            S_data.append(susceptible / len(df))
            I_data.append(infected / len(df))
            R_data.append(recovered / len(df))
    
    return {
        "steps": list(steps),
        "S": S_data,
        "I": I_data,
        "R": R_data
    }

# 基于网络的SIR模型
def network_sir_model(G, beta, gamma, initial_infected_nodes, steps, seed=42):
    """
    基于网络结构的SIR模型
    
    参数:
    G: NetworkX图对象，表示社交网络
    beta: 传播率
    gamma: 恢复率
    initial_infected_nodes: 初始感染节点列表
    steps: 运行步数
    seed: 随机种子
    
    返回:
    S, I, R时间序列
    """
    np.random.seed(seed)
    
    N = G.number_of_nodes()
    
    # 初始化状态 (0=S, 1=I, 2=R)
    states = np.zeros(N)
    states[initial_infected_nodes] = 1
    
    # 记录时间序列
    S_history = [sum(states == 0) / N]
    I_history = [sum(states == 1) / N]
    R_history = [sum(states == 2) / N]
    
    # 运行模拟
    for _ in range(steps):
        new_states = states.copy()
        
        # 对每个节点
        for node in range(N):
            # 如果节点是易感者(S)
            if states[node] == 0:
                # 查找感染邻居
                infected_neighbors = [neigh for neigh in G.neighbors(node) if states[neigh] == 1]
                
                # 计算感染概率
                if infected_neighbors:
                    p_infection = 1 - (1 - beta) ** len(infected_neighbors)
                    if np.random.random() < p_infection:
                        new_states[node] = 1  # 变为感染者
            
            # 如果节点是感染者(I)
            elif states[node] == 1:
                # 计算恢复概率
                if np.random.random() < gamma:
                    new_states[node] = 2  # 变为恢复者
        
        # 更新状态
        states = new_states
        
        # 记录历史
        S_history.append(sum(states == 0) / N)
        I_history.append(sum(states == 1) / N)
        R_history.append(sum(states == 2) / N)
    
    return {"S": S_history, "I": I_history, "R": R_history}

# 从LLM数据中估计SIR参数
def estimate_sir_parameters(llm_data):
    """
    从LLM数据中估计SIR模型的参数
    
    使用曲线拟合方法估计beta和gamma
    """
    # 定义SIR微分方程
    def sir_equations(t, beta, gamma, S0, I0, R0):
        """SIR模型的解析解（近似）"""
        S = S0 * np.exp(-beta * I0 * t)
        R = R0 + gamma * I0 * t
        I = 1 - S - R
        return S, I, R
    
    # 将数据整理为拟合所需格式
    steps = np.array(llm_data["steps"])
    S_data = np.array(llm_data["S"])
    I_data = np.array(llm_data["I"])
    R_data = np.array(llm_data["R"])
    
    # 定义拟合函数
    def fit_function(t, beta, gamma):
        S0, I0, R0 = S_data[0], I_data[0], R_data[0]
        S, I, R = sir_equations(t, beta, gamma, S0, I0, R0)
        return np.concatenate((S, I, R))
    
    # 准备拟合数据
    ydata = np.concatenate((S_data, I_data, R_data))
    xdata = np.tile(steps, 3)
    
    # 执行拟合
    try:
        popt, pcov = curve_fit(fit_function, xdata, ydata, bounds=([0, 0], [1, 1]))
        beta_est, gamma_est = popt
    except:
        # 如果拟合失败，使用基于启发式的方法
        # 通过观察I的变化估计参数
        dI = np.diff(I_data)
        # beta估计：初始阶段I的最大增长率除以初始易感者比例
        beta_est = max(0.1, min(0.5, np.max(dI) / (S_data[0] * I_data[0]) if S_data[0] * I_data[0] > 0 else 0.2))
        # gamma估计：I的平均下降率
        gamma_est = max(0.05, min(0.3, -np.mean(dI[dI < 0]) / np.mean(I_data[:-1][dI < 0]) if np.any(dI < 0) and np.mean(I_data[:-1][dI < 0]) > 0 else 0.1))
    
    return beta_est, gamma_est

# 从用户数据创建社交网络
def create_social_network_from_users(user_data, connection_probability=0.1):
    """从用户数据创建社交网络"""
    G = nx.Graph()
    
    # 添加节点
    for i in range(len(user_data)):
        G.add_node(i)
    
    # 添加边（基于人格特质和教育背景的相似性）
    for i in range(len(user_data)):
        for j in range(i+1, len(user_data)):
            # 计算人格特质相似度
            trait_similarity = 0
            for trait in ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']:
                trait_diff = abs(user_data.iloc[i][trait] - user_data.iloc[j][trait])
                trait_similarity += (3 - trait_diff) / 3  # 归一化相似度
            
            trait_similarity /= 5  # 平均相似度
            
            # 教育背景相似度
            edu_similarity = 1 - abs(user_data.iloc[i]['education'] - user_data.iloc[j]['education']) / 4
            
            # 综合相似度
            similarity = 0.7 * trait_similarity + 0.3 * edu_similarity
            
            # 根据相似度和基础概率决定是否连接
            if np.random.random() < connection_probability * (1 + similarity):
                G.add_edge(i, j)
    
    return G

# 直接在图表绘制过程中显式指定字体
def set_chinese_font():
    """设置支持中文的字体"""
    import platform
    system = platform.system()
    
    if system == "Darwin":  # macOS
        return {'family': 'Heiti TC', 'size': 12}
    elif system == "Windows":
        return {'family': 'SimHei', 'size': 12}
    else:  # Linux
        return {'family': 'WenQuanYi Micro Hei', 'size': 12}

# 获取适合当前系统的中文字体
chinese_font = set_chinese_font()

# 运行并比较两种模型
def compare_models(llm_data_file, n_simulations=10):
    """运行并比较LLM和传统SIR模型"""
    # 加载LLM结果
    llm_data = load_llm_results(llm_data_file)
    
    # 加载用户数据（用于创建网络）
    user_data = pd.read_csv(llm_data_file)
    
    # 创建社交网络
    G = create_social_network_from_users(user_data)
    
    # 估计SIR参数
    beta, gamma = estimate_sir_parameters(llm_data)
    print(f"估计的SIR参数: beta={beta:.4f}, gamma={gamma:.4f}")
    
    # 获取初始感染节点
    initial_infected = user_data[user_data['initial_belief'] == 1].index.tolist()
    
    # 运行多次传统SIR模拟并取平均值
    all_sir_results = []
    for i in range(n_simulations):
        sir_result = network_sir_model(
            G, 
            beta, 
            gamma, 
            initial_infected, 
            len(llm_data["steps"]) - 1,  # 减1是因为初始状态已经包含
            seed=42+i
        )
        all_sir_results.append(sir_result)
    
    # 计算平均值
    avg_sir = {
        "S": np.mean([r["S"] for r in all_sir_results], axis=0),
        "I": np.mean([r["I"] for r in all_sir_results], axis=0),
        "R": np.mean([r["R"] for r in all_sir_results], axis=0)
    }
    
    # 计算标准差（用于置信区间）
    std_sir = {
        "S": np.std([r["S"] for r in all_sir_results], axis=0),
        "I": np.std([r["I"] for r in all_sir_results], axis=0),
        "R": np.std([r["R"] for r in all_sir_results], axis=0)
    }
    
    # 绘制比较图
    plt.figure(figsize=(15, 10))
    
    # 设置风格
    sns.set_style("whitegrid")
    
    # 绘制LLM结果
    plt.plot(llm_data["steps"], llm_data["S"], 'b-', linewidth=2, label='LLM: Susceptible (S)')
    plt.plot(llm_data["steps"], llm_data["I"], 'r-', linewidth=2, label='LLM: Infected (I)')
    plt.plot(llm_data["steps"], llm_data["R"], 'g-', linewidth=2, label='LLM: Recovered (R)')
    
    # 绘制SIR模型结果（带置信区间）
    x = list(range(len(avg_sir["S"])))
    
    # S曲线
    plt.plot(x, avg_sir["S"], 'b--', linewidth=1.5, label='SIR: Susceptible (S)')
    plt.fill_between(x, avg_sir["S"] - std_sir["S"], avg_sir["S"] + std_sir["S"], color='b', alpha=0.2)
    
    # I曲线
    plt.plot(x, avg_sir["I"], 'r--', linewidth=1.5, label='SIR: Infected (I)')
    plt.fill_between(x, avg_sir["I"] - std_sir["I"], avg_sir["I"] + std_sir["I"], color='r', alpha=0.2)
    
    # R曲线
    plt.plot(x, avg_sir["R"], 'g--', linewidth=1.5, label='SIR: Recovered (R)')
    plt.fill_between(x, avg_sir["R"] - std_sir["R"], avg_sir["R"] + std_sir["R"], color='g', alpha=0.2)
    
    # 使用英文标题和标签，避免中文渲染问题
    plt.title('LLM Interactive Model vs Traditional SIR Model', fontsize=16)
    plt.xlabel('Time Steps', fontsize=14)
    plt.ylabel('Population Ratio', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图像
    plt.savefig(os.path.join('analysis_results/goal3.3', 'llm_vs_sir_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 计算差异指标
    differences = {
        "峰值感染者比例": {
            "LLM": max(llm_data["I"]),
            "SIR": max(avg_sir["I"]),
            "差异": max(llm_data["I"]) - max(avg_sir["I"])
        },
        "峰值时间": {
            "LLM": llm_data["I"].index(max(llm_data["I"])),
            "SIR": np.argmax(avg_sir["I"]),
            "差异": llm_data["I"].index(max(llm_data["I"])) - np.argmax(avg_sir["I"])
        },
        "最终感染者比例": {
            "LLM": llm_data["I"][-1],
            "SIR": avg_sir["I"][-1],
            "差异": llm_data["I"][-1] - avg_sir["I"][-1]
        },
        "最终恢复者比例": {
            "LLM": llm_data["R"][-1],
            "SIR": avg_sir["R"][-1],
            "差异": llm_data["R"][-1] - avg_sir["R"][-1]
        }
    }
    
    # 计算MSE
    mse = {
        "S": np.mean((np.array(llm_data["S"]) - avg_sir["S"][:len(llm_data["S"])])**2),
        "I": np.mean((np.array(llm_data["I"]) - avg_sir["I"][:len(llm_data["I"])])**2),
        "R": np.mean((np.array(llm_data["R"]) - avg_sir["R"][:len(llm_data["R"])])**2)
    }
    
    # 返回比较结果
    return {
        "llm_data": llm_data,
        "sir_data": avg_sir,
        "beta": beta,
        "gamma": gamma,
        "differences": differences,
        "mse": mse,
        "network": G
    }

# 主函数
if __name__ == "__main__":
    # 分析LLM数据和SIR模型
    results = compare_models("analysis_results/goal1.3/personality_analysis_data.csv")
    
    # 输出比较结果
    print("\n模型比较结果:")
    for metric, values in results["differences"].items():
        print(f"{metric}:")
        for model, value in values.items():
            print(f"  {model}: {value:.4f}")
    
    print("\n均方误差 (MSE):")
    for state, value in results["mse"].items():
        print(f"  {state}: {value:.6f}")
    
    # 网络统计信息
    G = results["network"]
    print(f"\n社交网络信息:")
    print(f"  节点数: {G.number_of_nodes()}")
    print(f"  边数: {G.number_of_edges()}")
    print(f"  平均度: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")
    print(f"  网络密度: {nx.density(G):.4f}")
    
    # 绘制网络结构图
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, node_size=50, node_color='lightblue', 
            edge_color='gray', alpha=0.7, width=0.5)
    plt.title("Social Network Structure Used in Simulation")
    plt.savefig(os.path.join('analysis_results/goal3.3', 'social_network.png'), dpi=300)
    plt.close()
    
    # 将分析结果保存到CSV文件
    # 创建数据帧保存比较结果
    comparison_data = {
        'Metric': [],
        'LLM_Value': [],
        'SIR_Value': [],
        'Difference': []
    }
    
    for metric, values in results["differences"].items():
        comparison_data['Metric'].append(metric)
        comparison_data['LLM_Value'].append(values['LLM'])
        comparison_data['SIR_Value'].append(values['SIR'])
        comparison_data['Difference'].append(values['差异'])
    
    # 添加MSE信息
    for state, value in results["mse"].items():
        comparison_data['Metric'].append(f'MSE_{state}')
        comparison_data['LLM_Value'].append(None)
        comparison_data['SIR_Value'].append(None)
        comparison_data['Difference'].append(value)
    
    # 创建数据帧并保存
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(os.path.join('analysis_results/goal3.3', 'model_comparison_results.csv'), index=False)
    
    # 保存传播参数
    params_df = pd.DataFrame({
        'Parameter': ['beta', 'gamma', 'R0'],
        'Value': [results['beta'], results['gamma'], results['beta']/results['gamma']]
    })
    params_df.to_csv(os.path.join('analysis_results/goal3.3', 'sir_parameters.csv'), index=False)
    
    # 保存传播曲线数据
    steps = list(range(len(results['llm_data']['S'])))
    curves_data = {
        'Time_Step': steps,
        'LLM_S': results['llm_data']['S'],
        'LLM_I': results['llm_data']['I'],
        'LLM_R': results['llm_data']['R'],
        'SIR_S': results['sir_data']['S'][:len(steps)],
        'SIR_I': results['sir_data']['I'][:len(steps)],
        'SIR_R': results['sir_data']['R'][:len(steps)]
    }
    curves_df = pd.DataFrame(curves_data)
    curves_df.to_csv(os.path.join('analysis_results/goal3.3', 'propagation_curves.csv'), index=False)
    
    print(f"\n分析完成。所有结果已保存到 'analysis_results/goal3.3' 文件夹")