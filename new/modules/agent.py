from typing import Dict, List, Any, Optional, Tuple
import uuid
import time
import os

from config_loader import get_config
from modules.memory_system import MemorySystem
from modules.belief_system import BeliefSystem
from utils.api_client import get_api_client

class Agent:
    """代理人类，代表一个参与者"""
    
    def __init__(self, 
                 name: str, 
                 age: int, 
                 traits: Dict[str, float],
                 health_status: str = "健康",
                 health_opinion: str = None,
                 agent_id: str = None):
        """初始化代理人
        
        Args:
            name: 代理人名称
            age: 代理人年龄
            traits: 代理人特质字典(大五人格特质)
            health_status: 健康状态，默认为"健康"
            health_opinion: 对健康话题的初始观点，如果为None则自动生成
            agent_id: 代理人ID，如果为None则自动生成
        """
        # 基本属性
        self.id = agent_id or str(uuid.uuid4())
        self.name = name
        self.age = age
        self.traits = traits
        self.health_status = health_status
        
        # 创建记忆系统
        self.memory_system = MemorySystem(self.id, self.name, self.traits, self.age)
        
        # 创建信念系统
        self.belief_system = BeliefSystem(self.id, self.name, self.traits, self.age)
        
        # 行为日志
        self.behaviors = []
        
        # 是否记录行为
        self.log_behaviors = get_config("agent.behavior_logging", False)
        
        # 初始化健康观点
        if health_opinion:
            self.memory_system.update_health_opinion(health_opinion)
            self.belief_system.set_belief("健康观点", health_opinion)
        else:
            self._generate_initial_health_opinion()
    
    def _generate_initial_health_opinion(self):
        """生成初始健康观点"""
        # 根据性格特质生成合适的健康观点
        api_client = get_api_client()
        
        # 格式化特质
        traits_desc = []
        for trait, value in self.traits.items():
            if value > 0.7:
                traits_desc.append(f"高{trait}")
            elif value < 0.3:
                traits_desc.append(f"低{trait}")
        traits_str = "、".join(traits_desc) if traits_desc else "普通性格"
        
        prompt = f"""
作为一名{self.age}岁的老年人，性格特点是{traits_str}，请生成一段关于健康（例如：营养、锻炼、医疗保健等）的个人观点。
这个观点应该反映出性格特点的影响。不超过50字。不需要解释，直接给出观点即可。
"""
        
        success, response = api_client.generate(prompt)
        if success:
            health_opinion = response["response"].strip()
            self.memory_system.update_health_opinion(health_opinion)
            self.belief_system.set_belief("健康观点", health_opinion)
        else:
            # 如果生成失败，使用默认观点
            default_opinion = "我认为保持健康的关键是均衡饮食和适量运动，定期体检也很重要。"
            self.memory_system.update_health_opinion(default_opinion)
            self.belief_system.set_belief("健康观点", default_opinion)
    
    def get_health_opinion(self) -> str:
        """获取当前健康观点
        
        Returns:
            健康观点
        """
        return self.memory_system.get_health_opinion()
    
    def update_health_opinion(self, new_opinion: str, reason: str = "通过交流更新"):
        """更新健康观点
        
        Args:
            new_opinion: 新的健康观点
            reason: 更新原因
        """
        self.memory_system.update_health_opinion(new_opinion)
        self.belief_system.set_belief("健康观点", new_opinion)
        
        if self.log_behaviors:
            self._log_behavior("更新健康观点", {
                "old_opinion": self.get_health_opinion(),
                "new_opinion": new_opinion,
                "reason": reason
            })
    
    def receive_information(self, information: str, source: str, category: str = "健康观点") -> Tuple[str, float, str]:
        """接收信息并可能更新信念
        
        Args:
            information: 信息内容
            source: 信息来源
            category: 信念类别
            
        Returns:
            (更新后的信念内容, 变化程度, 原因)
        """
        if self.log_behaviors:
            self._log_behavior("接收信息", {
                "information": information,
                "source": source,
                "category": category
            })
        
        # 处理新信息
        updated_belief, change_degree, reason = self.belief_system.process_new_information(
            category, information, source
        )
        
        # 如果变化显著，更新健康观点
        if category == "健康观点" and change_degree > 0.3:
            self.memory_system.update_health_opinion(updated_belief)
        
        return updated_belief, change_degree, reason
    
    def _log_behavior(self, behavior_type: str, details: Dict[str, Any]):
        """记录代理行为
        
        Args:
            behavior_type: 行为类型
            details: 行为详情
        """
        behavior = {
            "agent_id": self.id,
            "agent_name": self.name,
            "timestamp": time.time(),
            "type": behavior_type,
            "details": details
        }
        self.behaviors.append(behavior)
    
    def save_behaviors(self, output_dir: str = "output/behaviors"):
        """保存行为日志
        
        Args:
            output_dir: 输出目录
        """
        if not self.behaviors:
            return
        
        import json
        os.makedirs(output_dir, exist_ok=True)
        
        file_name = f"agent_{self.id}_behaviors.json"
        file_path = os.path.join(output_dir, file_name)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.behaviors, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存行为日志失败: {e}")
    
    def get_summary_short(self) -> str:
        """获取代理简短摘要
        
        Returns:
            代理简短摘要
        """
        return f"{self.name}，{self.age}岁，{self.health_status}"
    
    def get_summary_long(self) -> str:
        """获取代理详细摘要
        
        Returns:
            代理详细摘要
        """
        # 格式化特质
        traits_desc = []
        for trait, value in self.traits.items():
            if value > 0.7:
                traits_desc.append(f"高{trait}")
            elif value < 0.3:
                traits_desc.append(f"低{trait}")
        traits_str = "、".join(traits_desc) if traits_desc else "普通性格"
        
        health_opinion = self.get_health_opinion()
        
        return f"{self.name}，{self.age}岁，{self.health_status}。性格特点：{traits_str}。健康观点：{health_opinion}"
    
    def interact(self, other: 'Agent', topic: str = None) -> str:
        """与另一代理交互
        
        Args:
            other: 另一代理
            topic: 交互主题，如果为None则使用健康话题
            
        Returns:
            对话ID
        """
        from modules.dialogue_system import get_dialogue_manager
        dialogue_manager = get_dialogue_manager()
        
        if self.log_behaviors:
            self._log_behavior("开始交互", {
                "other_agent": other.id,
                "other_name": other.name,
                "topic": topic
            })
        
        dialogue_id = dialogue_manager.conduct_dialogue(self, other, topic)
        
        if self.log_behaviors:
            self._log_behavior("完成交互", {
                "other_agent": other.id,
                "other_name": other.name,
                "dialogue_id": dialogue_id
            })
        
        return dialogue_id
    
    def save(self, output_dir: str = "output/agents"):
        """保存代理状态
        
        Args:
            output_dir: 输出目录
        """
        import json
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存代理基本信息
        agent_data = {
            "id": self.id,
            "name": self.name,
            "age": self.age,
            "traits": self.traits,
            "health_status": self.health_status,
            "health_opinion": self.get_health_opinion()
        }
        
        file_name = f"agent_{self.id}.json"
        file_path = os.path.join(output_dir, file_name)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(agent_data, f, ensure_ascii=False, indent=2)
            
            # 保存记忆系统
            self.memory_system.save_memories()
            
            # 保存行为日志
            if self.log_behaviors:
                self.save_behaviors()
            
            return True
        except Exception as e:
            print(f"保存代理状态失败: {e}")
            return False
    
    @classmethod
    def load(cls, agent_id: str, output_dir: str = "output/agents") -> Optional['Agent']:
        """加载代理状态
        
        Args:
            agent_id: 代理ID
            output_dir: 输出目录
            
        Returns:
            代理对象，如果加载失败则返回None
        """
        import json
        
        file_name = f"agent_{agent_id}.json"
        file_path = os.path.join(output_dir, file_name)
        
        if not os.path.exists(file_path):
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                agent_data = json.load(f)
            
            # 创建代理
            agent = cls(
                name=agent_data["name"],
                age=agent_data["age"],
                traits=agent_data["traits"],
                health_status=agent_data["health_status"],
                health_opinion=agent_data["health_opinion"],
                agent_id=agent_data["id"]
            )
            
            return agent
        except Exception as e:
            print(f"加载代理状态失败: {e}")
            return None

# 创建代理工厂函数
def create_agent(name: str, age: int, traits: Dict[str, float] = None, 
                health_status: str = "健康", health_opinion: str = None) -> Agent:
    """创建代理
    
    Args:
        name: 代理名称
        age: 代理年龄
        traits: 代理特质字典，如果为None则随机生成
        health_status: 健康状态
        health_opinion: 健康观点
        
    Returns:
        代理对象
    """
    # 如果特质为None，则随机生成
    if traits is None:
        import random
        traits = {
            "外向性": random.uniform(0.0, 1.0),
            "宜人性": random.uniform(0.0, 1.0),
            "尽责性": random.uniform(0.0, 1.0),
            "神经质": random.uniform(0.0, 1.0),
            "开放性": random.uniform(0.0, 1.0)
        }
    
    return Agent(name, age, traits, health_status, health_opinion)