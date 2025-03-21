from typing import Dict, List, Any, Optional, Union, Tuple
import random
import os
import time
import pandas as pd
import pickle
import mesa
import networkx as nx

from config_loader import get_config
from modules.agent import Agent, create_agent
from modules.social_network import SocialNetwork, get_social_network
from modules.data_collection import setup_data_collector
from modules.dialogue_system import get_dialogue_manager
from utils.file_io import load_excel, ensure_dir, save_pickle

class World:
    """世界模型，整合各模块并进行模拟"""
    
    def __init__(self, 
                 initial_healthy: int = 100, 
                 initial_infected: int = 5, 
                 contact_rate: float = 0.3,
                 name: str = "健康谣言传播模拟",
                 days: int = 30,
                 user_data_file: str = None):
        """初始化世界模型
        
        Args:
            initial_healthy: 初始健康人数
            initial_infected: 初始感染人数
            contact_rate: 接触率
            name: 模拟名称
            days: 模拟天数
            user_data_file: 用户数据文件路径，如果提供则从文件加载真实用户数据
        """
        self.name = name
        self.initial_healthy = initial_healthy
        self.initial_infected = initial_infected
        self.total_population = initial_healthy + initial_infected
        self.contact_rate = contact_rate
        self.max_days = days
        
        # 当前状态
        self.day = 0
        self.step_count = 0
        self.is_running = False
        
        # 日计数器
        self.daily_new_cases = 0
        
        # 代理列表
        self.agents = []
        
        # 社交网络
        self.social_network = get_social_network()
        self.social_network.contact_rate = contact_rate
        
        # 对话管理器
        self.dialogue_manager = get_dialogue_manager()
        
        # 数据收集器
        self.data_collector = setup_data_collector(self.name)
        
        # 加载真实用户数据（如果提供）
        self.user_data = None
        if user_data_file and os.path.exists(user_data_file):
            try:
                self.user_data = load_excel(user_data_file)
                print(f"已加载{len(self.user_data)}条用户数据")
            except Exception as e:
                print(f"加载用户数据失败: {e}")
        
        # 初始化模型
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化模型"""
        # 创建代理
        self._create_agents()
        
        # 初始化社交网络
        agent_ids = [agent.id for agent in self.agents]
        self.social_network.generate_small_world_network(agent_ids)
        
        # 收集初始数据
        self.data_collector.collect(self)
    
    def _create_agents(self):
        """创建代理"""
        # 确定要创建的代理数量
        num_agents_to_create = self.total_population
        num_real_users = 0
        
        # 如果有用户数据，优先使用
        if self.user_data is not None and not self.user_data.empty:
            num_real_users = min(len(self.user_data), self.total_population)
            num_agents_to_create -= num_real_users
            
            # 创建真实用户代理
            for i in range(num_real_users):
                user_row = self.user_data.iloc[i]
                
                # 提取用户ID和名称
                user_id = str(user_row.get('user_id', f"user_{i}"))
                name = user_row.get('name', f"用户{i}")
                
                # 提取大五人格特质
                traits = {}
                for trait in ['外向性', '宜人性', '尽责性', '神经质', '开放性']:
                    if trait in user_row:
                        traits[trait] = float(user_row[trait])
                    else:
                        # 如果没有特质数据，使用随机值
                        traits[trait] = random.uniform(0.0, 1.0)
                
                # 提取健康观点
                health_opinion = None
                if 'health_opinion' in user_row and pd.notna(user_row['health_opinion']):
                    health_opinion = str(user_row['health_opinion'])
                
                # 创建代理
                agent = create_agent(
                    name=name,
                    age=random.randint(60, 85),  # 老年人年龄范围
                    traits=traits,
                    health_status="易感染",  # 初始为易感染状态
                    health_opinion=health_opinion
                )
                
                # 添加到代理列表
                self.agents.append(agent)
                
                # 添加到社交网络
                self.social_network.add_agent(agent.id, {"name": agent.name})
        
        # 创建随机代理补充剩余数量
        for i in range(num_agents_to_create):
            agent = create_agent(
                name=f"代理{num_real_users + i}",
                age=random.randint(60, 85),
                health_status="易感染"
            )
            
            # 添加到代理列表
            self.agents.append(agent)
            
            # 添加到社交网络
            self.social_network.add_agent(agent.id, {"name": agent.name})
        
        # 随机选择初始感染者
        infected_indices = random.sample(range(len(self.agents)), min(self.initial_infected, len(self.agents)))
        for idx in infected_indices:
            self.agents[idx].health_status = "感染"
        
        print(f"已创建{len(self.agents)}个代理（包含{num_real_users}个真实用户）")
    
    def step(self):
        """执行一个时间步"""
        # 增加步数计数
        self.step_count += 1
        
        # 获取交互对
        interaction_pairs = self.social_network.get_interaction_pairs()
        
        # 处理交互
        for agent1_id, agent2_id in interaction_pairs:
            # 获取代理对象
            agent1 = self._get_agent_by_id(agent1_id)
            agent2 = self._get_agent_by_id(agent2_id)
            
            if agent1 is None or agent2 is None:
                continue
            
            # 执行交互
            dialogue_id = agent1.interact(agent2)
            
            # 收集交互数据
            self.data_collector.collect_interaction(
                agent1_id=agent1.id,
                agent2_id=agent2.id,
                dialogue_id=dialogue_id
            )
            
            # 处理疾病传播（SIR模型）
            self._process_disease_transmission(agent1, agent2)
        
        # 模拟一天结束
        if self.step_count % 4 == 0:  # 假设一天有4个时间步
            self._end_day()
        
        # 收集数据
        self.data_collector.collect(self)
    
    def _process_disease_transmission(self, agent1: Agent, agent2: Agent):
        """处理疾病传播
        
        Args:
            agent1: 代理1
            agent2: 代理2
        """
        # 简单SIR模型实现
        # 如果一方是感染者，另一方是易感染者，则可能传播
        if agent1.health_status == "感染" and agent2.health_status == "易感染":
            if random.random() < self.contact_rate:
                agent2.health_status = "感染"
                self.daily_new_cases += 1
        elif agent2.health_status == "感染" and agent1.health_status == "易感染":
            if random.random() < self.contact_rate:
                agent1.health_status = "感染"
                self.daily_new_cases += 1
    
    def _end_day(self):
        """处理一天结束时的事件"""
        # 增加天数计数
        self.day += 1
        
        # 处理恢复
        for agent in self.agents:
            if agent.health_status == "感染":
                # 假设感染者有10%的概率每天恢复
                if random.random() < 0.1:
                    agent.health_status = "恢复"
        
        # 收集代理信念数据
        for agent in self.agents:
            self.data_collector.collect_agent_belief(
                agent_id=agent.id,
                name=agent.name,
                belief=agent.get_health_opinion(),
                confidence=0.8  # 简化实现，实际应从信念系统获取
            )
        
        # 重置日计数器
        self.daily_new_cases = 0
        
        # 检查模拟是否结束
        if self.day >= self.max_days:
            self.is_running = False
    
    def _get_agent_by_id(self, agent_id: str) -> Optional[Agent]:
        """根据ID获取代理
        
        Args:
            agent_id: 代理ID
            
        Returns:
            代理对象，如果不存在则返回None
        """
        for agent in self.agents:
            if agent.id == agent_id:
                return agent
        return None
    
    def run(self, steps: int = None):
        """运行模拟
        
        Args:
            steps: 运行步数，如果为None则运行到结束
        """
        self.is_running = True
        
        if steps is None:
            # 运行直到结束
            while self.is_running and self.day < self.max_days:
                self.step()
        else:
            # 运行指定步数
            for _ in range(steps):
                if not self.is_running or self.day >= self.max_days:
                    break
                self.step()
                
        print(f"模拟已完成 {self.day} 天, {self.step_count} 步")
    
    def save_results(self):
        """保存模拟结果"""
        # 保存数据
        self.data_collector.save_all()
        
        # 保存代理状态
        for agent in self.agents:
            agent.save()
    
    def save_checkpoint(self, checkpoint_dir: str = "checkpoint"):
        """保存检查点
        
        Args:
            checkpoint_dir: 检查点目录
        """
        ensure_dir(checkpoint_dir)
        
        run_dir = os.path.join(checkpoint_dir, f"run-{self.data_collector.run_id}")
        ensure_dir(run_dir)
        
        checkpoint_path = os.path.join(run_dir, f"{self.name}-checkpoint.pkl")
        
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(self, f)
            print(f"检查点已保存到: {checkpoint_path}")
            return True
        except Exception as e:
            print(f"保存检查点失败: {e}")
            return False
    
    @classmethod
    def load_checkpoint(cls, checkpoint_path: str) -> Optional['World']:
        """加载检查点
        
        Args:
            checkpoint_path: 检查点文件路径
            
        Returns:
            世界模型对象，如果加载失败则返回None
        """
        try:
            with open(checkpoint_path, 'rb') as f:
                world = pickle.load(f)
            print(f"已加载检查点: {checkpoint_path}")
            return world
        except Exception as e:
            print(f"加载检查点失败: {e}")
            return None

def create_world(
    initial_healthy: int = 10,
    initial_infected: int = 5,
    contact_rate: float = 0.3,
    name: str = "健康谣言传播模拟",
    days: int = 30,
    user_data_file: str = None
) -> World:
    """创建世界模型
    
    Args:
        initial_healthy: 初始健康人数
        initial_infected: 初始感染人数
        contact_rate: 接触率
        name: 模拟名称
        days: 模拟天数
        user_data_file: 用户数据文件路径
        
    Returns:
        世界模型对象
    """
    return World(
        initial_healthy=initial_healthy,
        initial_infected=initial_infected,
        contact_rate=contact_rate,
        name=name,
        days=days,
        user_data_file=user_data_file
    )