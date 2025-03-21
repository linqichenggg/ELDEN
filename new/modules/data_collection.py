from typing import Dict, List, Any, Optional, Callable
import pandas as pd
import numpy as np
import time
import os
import json
from datetime import datetime

from config_loader import get_config
from utils.file_io import ensure_dir, save_json, save_csv

class DataCollector:
    """数据收集器，用于收集和分析模拟数据"""
    
    def __init__(self, model_name: str = "GABM"):
        """初始化数据收集器
        
        Args:
            model_name: 模型名称，用于生成输出文件名
        """
        self.model_name = model_name
        self.data = {}
        self.start_time = time.time()
        self.collectors = {}
        self.series_data = {}
        self.agent_beliefs = {}
        self.agent_interactions = []
        
        # 输出目录
        self.output_dir = get_config("paths.output_dir", "output")
        ensure_dir(self.output_dir)
        
        # 运行ID
        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = os.path.join(self.output_dir, f"run-{self.run_id}")
        ensure_dir(self.run_dir)
    
    def add_collector(self, name: str, collector_func: Callable):
        """添加数据收集函数
        
        Args:
            name: 收集器名称
            collector_func: 收集器函数，接收模型作为参数，返回要收集的数据
        """
        self.collectors[name] = collector_func
    
    def collect(self, model: Any):
        """收集当前模型状态数据
        
        Args:
            model: 模型对象
        """
        # 收集时间
        current_time = time.time() - self.start_time
        
        # 基本数据
        basic_data = {
            "time": current_time,
            "step": model.step_count,
            "day": model.day
        }
        
        # 调用所有收集器函数
        for name, collector_func in self.collectors.items():
            try:
                data = collector_func(model)
                basic_data[name] = data
            except Exception as e:
                print(f"收集器 {name} 失败: {e}")
        
        # 保存数据点
        step_key = str(model.step_count)
        self.data[step_key] = basic_data
        
        # 更新时间序列数据
        for key, value in basic_data.items():
            if key not in ["time", "step"]:
                if key not in self.series_data:
                    self.series_data[key] = []
                self.series_data[key].append(value)
    
    def collect_agent_belief(self, agent_id: str, name: str, belief: str, confidence: float):
        """收集代理信念数据
        
        Args:
            agent_id: 代理ID
            name: 代理名称
            belief: 信念内容
            confidence: 信念确信度
        """
        if agent_id not in self.agent_beliefs:
            self.agent_beliefs[agent_id] = []
        
        self.agent_beliefs[agent_id].append({
            "timestamp": time.time(),
            "name": name,
            "belief": belief,
            "confidence": confidence
        })
    
    def collect_interaction(self, agent1_id: str, agent2_id: str, dialogue_id: str, 
                          topic: Optional[str] = None, outcome: Optional[Dict[str, Any]] = None):
        """收集代理互动数据
        
        Args:
            agent1_id: 代理1 ID
            agent2_id: 代理2 ID
            dialogue_id: 对话ID
            topic: 对话主题
            outcome: 对话结果
        """
        self.agent_interactions.append({
            "timestamp": time.time(),
            "agent1_id": agent1_id,
            "agent2_id": agent2_id,
            "dialogue_id": dialogue_id,
            "topic": topic,
            "outcome": outcome
        })
    
    def get_time_series_data(self) -> pd.DataFrame:
        """获取时间序列数据
        
        Returns:
            包含时间序列数据的DataFrame
        """
        # 确保所有序列长度相同
        max_len = max([len(series) for series in self.series_data.values()], default=0)
        
        # 填充缺失数据
        data = {}
        for key, series in self.series_data.items():
            if len(series) < max_len:
                # 用最后一个值填充
                last_value = series[-1] if series else None
                series = series + [last_value] * (max_len - len(series))
            data[key] = series
        
        # 创建DataFrame
        df = pd.DataFrame(data)
        return df
    
    def save_data_csv(self, file_name: Optional[str] = None):
        """保存数据为CSV文件
        
        Args:
            file_name: 文件名，如果为None则使用默认名称
        """
        df = self.get_time_series_data()
        
        if file_name is None:
            file_name = f"{self.model_name}-data.csv"
        
        file_path = os.path.join(self.run_dir, file_name)
        save_csv(df, file_path)
        print(f"数据已保存到: {file_path}")
    
    def save_agent_beliefs(self, file_name: Optional[str] = None):
        """保存代理信念数据为JSON文件
        
        Args:
            file_name: 文件名，如果为None则使用默认名称
        """
        if not self.agent_beliefs:
            return
        
        if file_name is None:
            file_name = f"{self.model_name}-agent-beliefs.json"
        
        file_path = os.path.join(self.run_dir, file_name)
        save_json(self.agent_beliefs, file_path)
        print(f"代理信念数据已保存到: {file_path}")
    
    def save_interactions(self, file_name: Optional[str] = None):
        """保存互动数据为JSON文件
        
        Args:
            file_name: 文件名，如果为None则使用默认名称
        """
        if not self.agent_interactions:
            return
        
        if file_name is None:
            file_name = f"{self.model_name}-interactions.json"
        
        file_path = os.path.join(self.run_dir, file_name)
        save_json(self.agent_interactions, file_path)
        print(f"互动数据已保存到: {file_path}")
    
    def save_all(self):
        """保存所有数据"""
        self.save_data_csv()
        self.save_agent_beliefs()
        self.save_interactions()
        
        # 保存原始数据
        raw_data_path = os.path.join(self.run_dir, f"{self.model_name}-raw-data.json")
        save_json(self.data, raw_data_path)
        
        print(f"所有数据已保存到目录: {self.run_dir}")

# 创建SIR模型相关的数据收集函数
def count_susceptible(model) -> int:
    """计算易感染个体数量
    
    Args:
        model: 模型对象
        
    Returns:
        易感染个体数量
    """
    return sum(1 for agent in model.agents if agent.health_status == "易感染")

def count_infected(model) -> int:
    """计算感染个体数量
    
    Args:
        model: 模型对象
        
    Returns:
        感染个体数量
    """
    return sum(1 for agent in model.agents if agent.health_status == "感染")

def count_recovered(model) -> int:
    """计算恢复个体数量
    
    Args:
        model: 模型对象
        
    Returns:
        恢复个体数量
    """
    return sum(1 for agent in model.agents if agent.health_status == "恢复")

def count_rumor_believers(model) -> int:
    """计算相信谣言的个体数量
    
    Args:
        model: 模型对象
        
    Returns:
        相信谣言的个体数量
    """
    # 简单实现：在实际项目中需要根据具体信念判断
    return 0  # 此处需要根据实际项目实现

def setup_data_collector(model_name: str = "GABM") -> DataCollector:
    """设置数据收集器
    
    Args:
        model_name: 模型名称
        
    Returns:
        数据收集器对象
    """
    collector = DataCollector(model_name)
    
    # 添加SIR模型数据收集器
    collector.add_collector("S", count_susceptible)
    collector.add_collector("I", count_infected)
    collector.add_collector("R", count_recovered)
    collector.add_collector("Believers", count_rumor_believers)
    
    return collector