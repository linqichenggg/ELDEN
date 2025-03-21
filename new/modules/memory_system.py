from typing import Dict, List, Any, Optional, Tuple
import json
import time
import datetime
import uuid
from collections import deque
import numpy as np
import os

from config_loader import get_config
from utils.api_client import get_api_client
from utils.prompt_templates import get_prompt_manager

class Memory:
    """记忆基类"""
    
    def __init__(self, content: str, source: str, importance: float = 0.5):
        """初始化记忆
        
        Args:
            content: 记忆内容
            source: 记忆来源（例如："对话"、"反思"）
            importance: 记忆重要性（0.0-1.0）
        """
        self.id = str(uuid.uuid4())
        self.content = content
        self.source = source
        self.importance = importance
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.access_count = 0
    
    def access(self):
        """访问记忆，更新访问时间和计数"""
        self.last_accessed = time.time()
        self.access_count += 1
    
    def get_age(self) -> float:
        """获取记忆年龄（秒）
        
        Returns:
            记忆年龄（秒）
        """
        return time.time() - self.created_at
    
    def get_recency(self) -> float:
        """获取记忆新近度（0.0-1.0，1.0表示最近）
        
        Returns:
            记忆新近度
        """
        # 使用指数衰减函数，半衰期为1天
        age_in_days = self.get_age() / (60 * 60 * 24)
        decay_rate = 0.5  # 每天衰减50%
        return np.exp(-decay_rate * age_in_days)
    
    def to_dict(self) -> Dict[str, Any]:
        """将记忆转换为字典
        
        Returns:
            记忆字典
        """
        return {
            "id": self.id,
            "content": self.content,
            "source": self.source,
            "importance": self.importance,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        """从字典创建记忆
        
        Args:
            data: 记忆字典
            
        Returns:
            记忆对象
        """
        memory = cls(data["content"], data["source"], data["importance"])
        memory.id = data["id"]
        memory.created_at = data["created_at"]
        memory.last_accessed = data["last_accessed"]
        memory.access_count = data["access_count"]
        return memory


class EpisodicMemory(Memory):
    """情节记忆，用于存储特定事件"""
    
    def __init__(self, content: str, source: str, 
                 participants: List[str] = None, 
                 location: str = None,
                 importance: float = 0.5):
        """初始化情节记忆
        
        Args:
            content: 记忆内容
            source: 记忆来源
            participants: 参与者列表
            location: 地点
            importance: 记忆重要性（0.0-1.0）
        """
        super().__init__(content, source, importance)
        self.participants = participants or []
        self.location = location
    
    def to_dict(self) -> Dict[str, Any]:
        """将记忆转换为字典
        
        Returns:
            记忆字典
        """
        data = super().to_dict()
        data.update({
            "type": "episodic",
            "participants": self.participants,
            "location": self.location
        })
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EpisodicMemory':
        """从字典创建记忆
        
        Args:
            data: 记忆字典
            
        Returns:
            记忆对象
        """
        memory = super().from_dict(data)
        memory.participants = data.get("participants", [])
        memory.location = data.get("location")
        return memory


class SemanticMemory(Memory):
    """语义记忆，用于存储知识和概念"""
    
    def __init__(self, content: str, source: str, 
                 category: str = None,
                 importance: float = 0.5):
        """初始化语义记忆
        
        Args:
            content: 记忆内容
            source: 记忆来源
            category: 类别
            importance: 记忆重要性（0.0-1.0）
        """
        super().__init__(content, source, importance)
        self.category = category
    
    def to_dict(self) -> Dict[str, Any]:
        """将记忆转换为字典
        
        Returns:
            记忆字典
        """
        data = super().to_dict()
        data.update({
            "type": "semantic",
            "category": self.category
        })
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SemanticMemory':
        """从字典创建记忆
        
        Args:
            data: 记忆字典
            
        Returns:
            记忆对象
        """
        memory = super().from_dict(data)
        memory.category = data.get("category")
        return memory


class MemorySystem:
    """记忆系统，管理短期和长期记忆"""
    
    def __init__(self, agent_id: str, agent_name: str, agent_traits: Dict[str, float], agent_age: int):
        """初始化记忆系统
        
        Args:
            agent_id: 代理ID
            agent_name: 代理名称
            agent_traits: 代理特质
            agent_age: 代理年龄
        """
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.agent_traits = agent_traits
        self.agent_age = agent_age
        
        # 短期记忆（最近对话）
        self.short_term_capacity = get_config("agent.memory.short_term_capacity", 10)
        self.short_term = deque(maxlen=self.short_term_capacity)
        
        # 工作记忆（当前上下文）
        self.working_memory_capacity = get_config("agent.memory.working_memory_capacity", 5)
        self.working_memory = []
        
        # 长期记忆
        self.long_term_episodic = []  # 情节记忆
        self.long_term_semantic = []  # 语义记忆（知识、信念）
        
        # API客户端
        self.api_client = get_api_client()
        
        # 提示模板管理器
        self.prompt_manager = get_prompt_manager()
        
        # 加载记忆（如果有）
        self.load_memories()
    
    def add_to_short_term(self, content: str, source: str, metadata: Dict[str, Any] = None):
        """添加到短期记忆
        
        Args:
            content: 记忆内容
            source: 记忆来源
            metadata: 元数据
        """
        memory = {
            "content": content,
            "source": source,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        self.short_term.append(memory)
    
    def add_dialogue(self, speaker: str, content: str):
        """添加对话到短期记忆
        
        Args:
            speaker: 说话者
            content: 对话内容
        """
        self.add_to_short_term(content, "dialogue", {"speaker": speaker})
    
    def get_recent_dialogues(self, count: int = None) -> List[Dict[str, Any]]:
        """获取最近对话
        
        Args:
            count: 获取数量，None表示全部
            
        Returns:
            对话列表
        """
        dialogues = [m for m in self.short_term if m["source"] == "dialogue"]
        if count is not None:
            dialogues = dialogues[-count:]
        return dialogues
    
    def format_dialogue_history(self, count: int = None) -> str:
        """格式化对话历史
        
        Args:
            count: 获取数量，None表示全部
            
        Returns:
            格式化的对话历史
        """
        dialogues = self.get_recent_dialogues(count)
        formatted = []
        for dialogue in dialogues:
            speaker = dialogue["metadata"].get("speaker", "未知")
            content = dialogue["content"]
            formatted.append(f"{speaker}: {content}")
        return "\n".join(formatted)
    
    def add_to_working_memory(self, memory: Memory):
        """添加到工作记忆
        
        Args:
            memory: 记忆对象
        """
        # 如果达到容量，移除重要性最低的记忆
        if len(self.working_memory) >= self.working_memory_capacity:
            self.working_memory.sort(key=lambda m: m.importance)
            self.working_memory.pop(0)  # 移除重要性最低的
        
        self.working_memory.append(memory)
    
    def add_to_long_term(self, memory: Memory):
        """添加到长期记忆
        
        Args:
            memory: 记忆对象
        """
        if isinstance(memory, EpisodicMemory):
            self.long_term_episodic.append(memory)
        elif isinstance(memory, SemanticMemory):
            self.long_term_semantic.append(memory)
    
    def consolidate_dialogue_to_episodic(self, participants: List[str] = None, importance: float = None):
        """将对话整合为情节记忆
        
        Args:
            participants: 参与者列表
            importance: 重要性，如果为None则自动计算
        """
        if not self.short_term:
            return
        
        # 获取最近对话
        dialogues = self.get_recent_dialogues()
        if not dialogues:
            return
        
        # 生成对话摘要
        dialogue_history = self.format_dialogue_history()
        prompt = self.prompt_manager.format_prompt("memory_consolidation",
            agent_name=self.agent_name,
            agent_age=self.agent_age,
            agent_traits=self._format_traits(),
            memory_fragments=dialogue_history
        )
        
        success, response = self.api_client.generate(prompt)
        if success:
            summary = response["response"]
            
            # 如果未指定重要性，则根据内容自动计算
            if importance is None:
                # 基于情感强度、涉及健康话题的程度等因素计算重要性
                # 简单实现：对话越长，假设越重要
                importance = min(0.3 + len(dialogues) * 0.05, 0.9)
            
            # 创建情节记忆
            memory = EpisodicMemory(
                content=summary,
                source="对话整合",
                participants=participants or [],
                importance=importance
            )
            
            # 添加到长期记忆
            self.add_to_long_term(memory)
            return memory
        
        return None
    
    def add_belief(self, content: str, category: str = "健康观点", importance: float = 0.7):
        """添加信念（语义记忆）
        
        Args:
            content: 信念内容
            category: 类别
            importance: 重要性
            
        Returns:
            创建的记忆对象
        """
        memory = SemanticMemory(
            content=content,
            source="信念形成",
            category=category,
            importance=importance
        )
        
        # 检查是否已存在类似信念
        for i, belief in enumerate(self.long_term_semantic):
            if belief.category == category:
                # 替换旧信念
                self.long_term_semantic[i] = memory
                return memory
        
        # 添加新信念
        self.add_to_long_term(memory)
        return memory
    
    def get_beliefs(self, category: str = None) -> List[SemanticMemory]:
        """获取信念
        
        Args:
            category: 类别筛选
            
        Returns:
            信念列表
        """
        if category:
            return [b for b in self.long_term_semantic if b.category == category]
        return self.long_term_semantic
    
    def get_health_opinion(self) -> str:
        """获取健康观点
        
        Returns:
            健康观点
        """
        beliefs = self.get_beliefs("健康观点")
        if beliefs:
            # 获取最重要的健康观点
            beliefs.sort(key=lambda b: b.importance, reverse=True)
            return beliefs[0].content
        return "我对健康没有特别的看法。"
    
    def update_health_opinion(self, new_opinion: str, importance: float = 0.7):
        """更新健康观点
        
        Args:
            new_opinion: 新观点
            importance: 重要性
        """
        self.add_belief(new_opinion, "健康观点", importance)
    
    def retrieve_relevant_memories(self, query: str, max_results: int = 5) -> List[Memory]:
        """检索相关记忆
        
        Args:
            query: 查询字符串
            max_results: 最大结果数
            
        Returns:
            相关记忆列表
        """
        # 简单实现：关键词匹配
        # 实际项目中可以使用向量数据库或嵌入相似度
        all_memories = self.long_term_episodic + self.long_term_semantic
        
        # 计算相关性分数
        results = []
        for memory in all_memories:
            # 简单相关性：内容包含查询词的片段
            relevance = 0
            if query.lower() in memory.content.lower():
                relevance = 0.8
            elif any(term in memory.content.lower() for term in query.lower().split()):
                relevance = 0.4
            
            # 考虑重要性和新近度
            if relevance > 0:
                score = relevance * 0.5 + memory.importance * 0.3 + memory.get_recency() * 0.2
                results.append((memory, score))
        
        # 排序并选择最相关的
        results.sort(key=lambda x: x[1], reverse=True)
        return [r[0] for r in results[:max_results]]
    
    def _format_traits(self) -> str:
        """格式化代理特质
        
        Returns:
            格式化的特质字符串
        """
        traits_str = []
        for trait, value in self.agent_traits.items():
            if value > 0.7:
                traits_str.append(f"高{trait}")
            elif value < 0.3:
                traits_str.append(f"低{trait}")
        return "、".join(traits_str) if traits_str else "普通性格"
    
    def save_memories(self):
        """保存记忆到文件"""
        # 确保目录存在
        os.makedirs("data/memories", exist_ok=True)
        
        # 保存长期记忆
        memories = {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "episodic": [m.to_dict() for m in self.long_term_episodic],
            "semantic": [m.to_dict() for m in self.long_term_semantic]
        }
        
        file_path = f"data/memories/{self.agent_id}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(memories, f, ensure_ascii=False, indent=2)
    
    def load_memories(self):
        """从文件加载记忆"""
        file_path = f"data/memories/{self.agent_id}.json"
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 加载情节记忆
                for memory_data in data.get("episodic", []):
                    memory = EpisodicMemory.from_dict(memory_data)
                    self.long_term_episodic.append(memory)
                
                # 加载语义记忆
                for memory_data in data.get("semantic", []):
                    memory = SemanticMemory.from_dict(memory_data)
                    self.long_term_semantic.append(memory)
                
                print(f"已为{self.agent_name}加载{len(self.long_term_episodic)}条情节记忆和{len(self.long_term_semantic)}条语义记忆")
            except Exception as e:
                print(f"加载记忆失败: {e}")