import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx
import os
import json
from typing import Dict, List, Any, Optional, Tuple
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime

from config_loader import get_config
from utils.file_io import ensure_dir

class Visualizer:
    """可视化工具类，用于绘制各种图表和网络"""
    
    def __init__(self, output_dir: str = None):
        """初始化可视化工具
        
        Args:
            output_dir: 输出目录，如果为None则使用配置值
        """
        self.output_dir = output_dir or get_config("paths.output_dir", "output")
        ensure_dir(self.output_dir)
        
        # 设置绘图样式
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("Set2")
    
    def plot_sir_curve(self, data: pd.DataFrame, title: str = "SIR模型", 
                       filename: str = "sir_curve.png"):
        """绘制SIR曲线
        
        Args:
            data: 包含S、I、R数据的DataFrame
            title: 图表标题
            filename: 输出文件名
        """
        plt.figure(figsize=(10, 6))
        
        # 绘制SIR曲线
        if 'S' in data.columns:
            plt.plot(data['S'], label='易感染', color='#3498db', linewidth=2)
        if 'I' in data.columns:
            plt.plot(data['I'], label='感染', color='#e74c3c', linewidth=2)
        if 'R' in data.columns:
            plt.plot(data['R'], label='恢复', color='#2ecc71', linewidth=2)
        
        # 设置图表
        plt.title(title, fontsize=16)
        plt.xlabel('时间步', fontsize=12)
        plt.ylabel('人数', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 添加边框
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#888888')
        
        # 保存图表
        file_path = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(file_path, dpi=300)
        plt.close()
        
        print(f"SIR曲线已保存到: {file_path}")
    
    def plot_rumor_spread(self, data: pd.DataFrame, title: str = "谣言传播", 
                         filename: str = "rumor_spread.png"):
        """绘制谣言传播曲线
        
        Args:
            data: 包含Believers数据的DataFrame
            title: 图表标题
            filename: 输出文件名
        """
        plt.figure(figsize=(10, 6))
        
        # 绘制谣言传播曲线
        if 'Believers' in data.columns:
            plt.plot(data['Believers'], label='相信谣言', color='#9b59b6', linewidth=2)
        
        # 设置图表
        plt.title(title, fontsize=16)
        plt.xlabel('时间步', fontsize=12)
        plt.ylabel('人数', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 添加边框
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#888888')
        
        # 保存图表
        file_path = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(file_path, dpi=300)
        plt.close()
        
        print(f"谣言传播曲线已保存到: {file_path}")
    
    def plot_network(self, network: nx.Graph, title: str = "社交网络", 
                     filename: str = "social_network.png",
                     node_color_map: Dict[str, str] = None):
        """绘制社交网络
        
        Args:
            network: NetworkX图对象
            title: 图表标题
            filename: 输出文件名
            node_color_map: 节点颜色映射，格式为{节点ID: 颜色}
        """
        plt.figure(figsize=(12, 12))
        
        # 设置节点位置
        pos = nx.spring_layout(network, seed=42)
        
        # 设置节点颜色
        if node_color_map:
            node_colors = [node_color_map.get(node, '#3498db') for node in network.nodes()]
        else:
            node_colors = '#3498db'
        
        # 绘制网络
        nx.draw_networkx(
            network,
            pos=pos,
            with_labels=False,
            node_color=node_colors,
            node_size=100,
            edge_color='#bbbbbb',
            width=0.5,
            alpha=0.8
        )
        
        # 设置图表
        plt.title(title, fontsize=16)
        plt.axis('off')
        
        # 保存图表
        file_path = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(file_path, dpi=300)
        plt.close()
        
        print(f"社交网络图已保存到: {file_path}")
    
    def plot_belief_distribution(self, beliefs: Dict[str, List[Dict[str, Any]]], 
                               title: str = "信念分布", 
                               filename: str = "belief_distribution.png"):
        """绘制信念分布热图
        
        Args:
            beliefs: 代理信念数据，格式为{agent_id: [{belief信息},...]}
            title: 图表标题
            filename: 输出文件名
        """
        # 提取最后一次信念记录
        latest_beliefs = {}
        for agent_id, agent_beliefs in beliefs.items():
            if agent_beliefs:
                # 按时间戳排序
                sorted_beliefs = sorted(agent_beliefs, key=lambda x: x.get('timestamp', 0))
                latest_beliefs[agent_id] = sorted_beliefs[-1]
        
        # 如果没有足够的数据，则跳过
        if len(latest_beliefs) < 2:
            print("没有足够的信念数据用于可视化")
            return
        
        # 创建数据
        agent_ids = list(latest_beliefs.keys())
        confidence_values = [b.get('confidence', 0.5) for b in latest_beliefs.values()]
        
        plt.figure(figsize=(12, 6))
        
        # 绘制条形图
        plt.bar(range(len(agent_ids)), confidence_values, color=plt.cm.viridis(np.array(confidence_values)))
        
        # 设置图表
        plt.title(title, fontsize=16)
        plt.xlabel('代理', fontsize=12)
        plt.ylabel('信念确信度', fontsize=12)
        plt.xticks([])  # 隐藏x轴标签
        plt.ylim(0, 1)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 添加边框
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#888888')
        
        # 添加颜色条
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label('确信度', fontsize=12)
        
        # 保存图表
        file_path = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(file_path, dpi=300)
        plt.close()
        
        print(f"信念分布图已保存到: {file_path}")
    
    def plot_interaction_heatmap(self, interactions: List[Dict[str, Any]], 
                               title: str = "互动热图", 
                               filename: str = "interaction_heatmap.png"):
        """绘制互动热图
        
        Args:
            interactions: 互动数据列表
            title: 图表标题
            filename: 输出文件名
        """
        # 提取代理ID
        agent_ids = set()
        for interaction in interactions:
            agent_ids.add(interaction.get('agent1_id', ''))
            agent_ids.add(interaction.get('agent2_id', ''))
        
        agent_ids = list(agent_ids)
        n_agents = len(agent_ids)
        
        # 如果没有足够的数据，则跳过
        if n_agents < 2:
            print("没有足够的互动数据用于可视化")
            return
        
        # 创建互动矩阵
        interaction_matrix = np.zeros((n_agents, n_agents))
        
        # 填充互动矩阵
        for interaction in interactions:
            agent1 = interaction.get('agent1_id', '')
            agent2 = interaction.get('agent2_id', '')
            
            if agent1 in agent_ids and agent2 in agent_ids:
                i = agent_ids.index(agent1)
                j = agent_ids.index(agent2)
                interaction_matrix[i, j] += 1
                interaction_matrix[j, i] += 1  # 双向计数
        
        plt.figure(figsize=(10, 8))
        
        # 创建热图
        sns.heatmap(
            interaction_matrix,
            cmap='YlOrRd',
            square=True,
            linewidths=.5,
            cbar_kws={"shrink": .8},
            annot=False,
            fmt='d'
        )
        
        # 设置图表
        plt.title(title, fontsize=16)
        plt.xlabel('代理', fontsize=12)
        plt.ylabel('代理', fontsize=12)
        
        # 简化标签
        n_ticks = min(10, n_agents)
        tick_indices = np.linspace(0, n_agents - 1, n_ticks, dtype=int)
        plt.xticks(tick_indices + 0.5, [f"A{i}" for i in tick_indices])
        plt.yticks(tick_indices + 0.5, [f"A{i}" for i in tick_indices])
        
        # 保存图表
        file_path = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        plt.savefig(file_path, dpi=300)
        plt.close()
        
        print(f"互动热图已保存到: {file_path}")
    
    def visualize_all(self, data_file: str, beliefs_file: str = None, 
                    interactions_file: str = None, network_file: str = None):
        """生成所有可视化
        
        Args:
            data_file: 时间序列数据文件
            beliefs_file: 信念数据文件
            interactions_file: 互动数据文件
            network_file: 网络数据文件
        """
        # 加载时间序列数据
        try:
            data = pd.read_csv(data_file)
            
            # 绘制SIR曲线
            self.plot_sir_curve(data, "SIR模型传播曲线", "sir_curve.png")
            
            # 绘制谣言传播曲线
            if 'Believers' in data.columns:
                self.plot_rumor_spread(data, "谣言传播趋势", "rumor_spread.png")
        except Exception as e:
            print(f"加载时间序列数据失败: {e}")
        
        # 加载信念数据
        if beliefs_file and os.path.exists(beliefs_file):
            try:
                with open(beliefs_file, 'r', encoding='utf-8') as f:
                    beliefs = json.load(f)
                
                # 绘制信念分布
                self.plot_belief_distribution(beliefs, "代理信念确信度分布", "belief_distribution.png")
            except Exception as e:
                print(f"加载信念数据失败: {e}")
        
        # 加载互动数据
        if interactions_file and os.path.exists(interactions_file):
            try:
                with open(interactions_file, 'r', encoding='utf-8') as f:
                    interactions = json.load(f)
                
                # 绘制互动热图
                self.plot_interaction_heatmap(interactions, "代理互动频率热图", "interaction_heatmap.png")
            except Exception as e:
                print(f"加载互动数据失败: {e}")
        
        # 加载网络数据
        if network_file and os.path.exists(network_file):
            try:
                network = nx.readwrite.json_graph.node_link_graph(
                    json.load(open(network_file, 'r', encoding='utf-8'))
                )
                
                # 绘制社交网络
                self.plot_network(network, "社交网络结构", "social_network.png")
            except Exception as e:
                print(f"加载网络数据失败: {e}")


def visualize_simulation_results(run_dir: str):
    """可视化模拟结果
    
    Args:
        run_dir: 运行结果目录
    """
    visualizer = Visualizer(run_dir)
    
    # 查找数据文件
    data_file = None
    beliefs_file = None
    interactions_file = None
    network_file = None
    
    for file in os.listdir(run_dir):
        if file.endswith('-data.csv'):
            data_file = os.path.join(run_dir, file)
        elif file.endswith('-agent-beliefs.json'):
            beliefs_file = os.path.join(run_dir, file)
        elif file.endswith('-interactions.json'):
            interactions_file = os.path.join(run_dir, file)
        elif file.endswith('-network.json'):
            network_file = os.path.join(run_dir, file)
    
    # 生成可视化
    if data_file:
        visualizer.visualize_all(data_file, beliefs_file, interactions_file, network_file)
    else:
        print(f"在目录 {run_dir} 中未找到数据文件")