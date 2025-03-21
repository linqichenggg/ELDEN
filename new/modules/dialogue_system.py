from typing import Dict, List, Any, Optional, Tuple, Callable
import time
import uuid
import json
import os

from config_loader import get_config
from utils.api_client import get_api_client
from utils.prompt_templates import get_prompt_manager

class DialogueManager:
    """对话系统，管理代理之间的对话"""
    
    def __init__(self):
        """初始化对话系统"""
        # 加载配置
        self.max_turns = get_config("dialogue.max_dialogue_turns", 10)
        self.convergence_threshold = get_config("dialogue.convergence_threshold", 0.8)
        self.save_dialogues = get_config("dialogue.save_dialogues", True)
        
        # API客户端
        self.api_client = get_api_client()
        
        # 提示模板管理器
        self.prompt_manager = get_prompt_manager()
        
        # 当前活跃对话
        self.active_dialogues = {}
    
    def create_dialogue(self, agent1_id: str, agent2_id: str) -> str:
        """创建新对话
        
        Args:
            agent1_id: 代理1 ID
            agent2_id: 代理2 ID
            
        Returns:
            对话ID
        """
        dialogue_id = str(uuid.uuid4())
        
        self.active_dialogues[dialogue_id] = {
            "id": dialogue_id,
            "agent1_id": agent1_id,
            "agent2_id": agent2_id,
            "turns": [],
            "state": "active",
            "created_at": time.time(),
            "last_updated": time.time()
        }
        
        return dialogue_id
    
    def add_message(self, dialogue_id: str, sender_id: str, content: str, 
                    metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """添加消息到对话
        
        Args:
            dialogue_id: 对话ID
            sender_id: 发送者ID
            content: 消息内容
            metadata: 元数据
            
        Returns:
            消息字典
        """
        if dialogue_id not in self.active_dialogues:
            raise ValueError(f"对话 {dialogue_id} 不存在")
        
        dialogue = self.active_dialogues[dialogue_id]
        
        # 验证发送者
        if sender_id != dialogue["agent1_id"] and sender_id != dialogue["agent2_id"]:
            raise ValueError(f"发送者 {sender_id} 不是对话的参与者")
        
        # 创建消息
        message = {
            "id": str(uuid.uuid4()),
            "dialogue_id": dialogue_id,
            "sender_id": sender_id,
            "receiver_id": dialogue["agent2_id"] if sender_id == dialogue["agent1_id"] else dialogue["agent1_id"],
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        # 添加到对话
        dialogue["turns"].append(message)
        dialogue["last_updated"] = time.time()
        
        # 如果达到最大轮数，将对话设为完成
        if len(dialogue["turns"]) >= self.max_turns:
            dialogue["state"] = "completed"
        
        return message
    
    def get_dialogue(self, dialogue_id: str) -> Optional[Dict[str, Any]]:
        """获取对话
        
        Args:
            dialogue_id: 对话ID
            
        Returns:
            对话字典，如果不存在则返回None
        """
        return self.active_dialogues.get(dialogue_id)
    
    def get_dialogue_messages(self, dialogue_id: str) -> List[Dict[str, Any]]:
        """获取对话消息
        
        Args:
            dialogue_id: 对话ID
            
        Returns:
            消息列表
        """
        dialogue = self.get_dialogue(dialogue_id)
        if not dialogue:
            return []
        return dialogue["turns"]
    
    def format_dialogue_history(self, dialogue_id: str, max_turns: int = None) -> str:
        """格式化对话历史
        
        Args:
            dialogue_id: 对话ID
            max_turns: 最大轮数，None表示全部
            
        Returns:
            格式化的对话历史
        """
        messages = self.get_dialogue_messages(dialogue_id)
        if max_turns is not None:
            messages = messages[-max_turns:]
        
        formatted = []
        for msg in messages:
            sender_id = msg["sender_id"]
            content = msg["content"]
            formatted.append(f"{sender_id}: {content}")
        
        return "\n".join(formatted)
    
    def generate_response(self, dialogue_id: str, agent_info: Dict[str, Any]) -> str:
        """生成代理回复
        
        Args:
            dialogue_id: 对话ID
            agent_info: 代理信息字典
                - name: 代理名称
                - age: 代理年龄
                - traits: 代理特质字典
                - id: 代理ID
                - health_opinion: 代理的健康观点
            
        Returns:
            生成的回复
        """
        dialogue = self.get_dialogue(dialogue_id)
        if not dialogue or dialogue["state"] != "active":
            return ""
        
        messages = self.get_dialogue_messages(dialogue_id)
        if not messages:
            return ""
        
        # 获取上一条消息
        last_message = messages[-1]
        
        # 确保回复者不是最后发言者
        if last_message["receiver_id"] != agent_info["id"]:
            return ""
        
        # 获取对方名称（简单实现）
        other_id = last_message["sender_id"]
        other_name = other_id  # 实际实现中应从agent管理器获取名称
        
        # 格式化代理特质
        traits_str = self._format_traits(agent_info["traits"])
        
        # 生成回复
        prompt = self.prompt_manager.format_prompt("agent_interaction",
            agent_name=agent_info["name"],
            agent_age=agent_info["age"],
            agent_traits=traits_str,
            other_name=other_name,
            message=last_message["content"],
            health_opinion=agent_info["health_opinion"]
        )
        
        success, response = self.api_client.generate(prompt)
        if success:
            return response["response"]
        
        return "我不太确定该怎么回应。"
    
    def get_dialogue_reflection(self, dialogue_id: str, agent_info: Dict[str, Any]) -> Dict[str, Any]:
        """获取对话反思
        
        Args:
            dialogue_id: 对话ID
            agent_info: 代理信息字典
                - name: 代理名称
                - age: 代理年龄
                - traits: 代理特质字典
                - id: 代理ID
                - health_opinion: 代理的健康观点
            
        Returns:
            反思结果字典，包含更新的观点和原因
        """
        dialogue = self.get_dialogue(dialogue_id)
        if not dialogue:
            return {"updated_opinion": agent_info["health_opinion"], "reason": "没有进行对话"}
        
        # 格式化对话历史
        dialogue_history = self.format_dialogue_history(dialogue_id)
        
        # 获取对方名称（简单实现）
        other_id = dialogue["agent1_id"] if agent_info["id"] == dialogue["agent2_id"] else dialogue["agent2_id"]
        other_name = other_id  # 实际实现中应从agent管理器获取名称
        
        # 格式化代理特质
        traits_str = self._format_traits(agent_info["traits"])
        
        # 生成反思
        prompt = self.prompt_manager.format_prompt("agent_reflection",
            agent_name=agent_info["name"],
            agent_age=agent_info["age"],
            agent_traits=traits_str,
            other_name=other_name,
            dialogue_history=dialogue_history,
            previous_opinion=agent_info["health_opinion"]
        )
        
        success, response = self.api_client.generate(prompt)
        if success:
            response_text = response["response"]
            
            # 解析响应获取更新后的观点和原因
            updated_opinion = agent_info["health_opinion"]  # 默认不变
            reason = "观点保持不变"
            
            # 简单解析（实际可能需要更复杂的解析）
            for line in response_text.split("\n"):
                if line.startswith("更新观点:"):
                    updated_opinion = line.replace("更新观点:", "").strip()
                elif line.startswith("原因:"):
                    reason = line.replace("原因:", "").strip()
            
            return {
                "updated_opinion": updated_opinion,
                "reason": reason
            }
        
        return {"updated_opinion": agent_info["health_opinion"], "reason": "反思生成失败"}
    
    def complete_dialogue(self, dialogue_id: str):
        """完成对话
        
        Args:
            dialogue_id: 对话ID
        """
        if dialogue_id in self.active_dialogues:
            self.active_dialogues[dialogue_id]["state"] = "completed"
            
            # 如果配置为保存对话，则保存
            if self.save_dialogues:
                self.save_dialogue(dialogue_id)
    
    def save_dialogue(self, dialogue_id: str):
        """保存对话到文件
        
        Args:
            dialogue_id: 对话ID
        """
        dialogue = self.get_dialogue(dialogue_id)
        if not dialogue:
            return
        
        os.makedirs("output/dialogues", exist_ok=True)
        
        file_name = f"dialogue_{dialogue_id}.json"
        file_path = os.path.join("output/dialogues", file_name)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(dialogue, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存对话失败: {e}")
    
    def save_all_dialogues(self):
        """保存所有对话"""
        for dialogue_id in self.active_dialogues:
            self.save_dialogue(dialogue_id)
    
    def _format_traits(self, traits: Dict[str, float]) -> str:
        """格式化代理特质
        
        Args:
            traits: 特质字典
            
        Returns:
            格式化的特质字符串
        """
        traits_str = []
        for trait, value in traits.items():
            if value > 0.7:
                traits_str.append(f"高{trait}")
            elif value < 0.3:
                traits_str.append(f"低{trait}")
        return "、".join(traits_str) if traits_str else "普通性格"
    
    def conduct_dialogue(self, agent1: Any, agent2: Any, topic: str = None, 
                         max_turns: int = None) -> str:
        """进行一次完整对话
        
        Args:
            agent1: 代理1对象
            agent2: 代理2对象
            topic: 对话主题，如果为None则使用健康谣言
            max_turns: 最大对话轮数，如果为None则使用配置值
            
        Returns:
            对话ID
        """
        if max_turns is None:
            max_turns = self.max_turns
        
        # 创建对话
        dialogue_id = self.create_dialogue(agent1.id, agent2.id)
        
        # 初始消息（如果有主题）
        if topic:
            initial_message = f"你好，我想和你讨论一下关于{topic}的问题。"
            self.add_message(dialogue_id, agent1.id, initial_message)
        
        # 进行对话
        for _ in range(max_turns):
            # 获取当前对话
            dialogue = self.get_dialogue(dialogue_id)
            if dialogue["state"] != "active":
                break
            
            messages = self.get_dialogue_messages(dialogue_id)
            
            # 确定下一个发言者
            if not messages:
                next_speaker = agent1
                other = agent2
            else:
                last_message = messages[-1]
                if last_message["sender_id"] == agent1.id:
                    next_speaker = agent2
                    other = agent1
                else:
                    next_speaker = agent1
                    other = agent2
            
            # 生成回复
            speaker_info = {
                "id": next_speaker.id,
                "name": next_speaker.name,
                "age": next_speaker.age,
                "traits": next_speaker.traits,
                "health_opinion": next_speaker.get_health_opinion()
            }
            
            response = self.generate_response(dialogue_id, speaker_info)
            if not response:
                break
            
            # 添加回复到对话
            self.add_message(dialogue_id, next_speaker.id, response)
            
            # 更新代理记忆
            next_speaker.memory_system.add_dialogue(other.name, messages[-1]["content"] if messages else "")
            next_speaker.memory_system.add_dialogue(next_speaker.name, response)
        
        # 对话结束后进行反思
        agent1_reflection = self.get_dialogue_reflection(dialogue_id, {
            "id": agent1.id,
            "name": agent1.name,
            "age": agent1.age,
            "traits": agent1.traits,
            "health_opinion": agent1.get_health_opinion()
        })
        
        agent2_reflection = self.get_dialogue_reflection(dialogue_id, {
            "id": agent2.id,
            "name": agent2.name,
            "age": agent2.age,
            "traits": agent2.traits,
            "health_opinion": agent2.get_health_opinion()
        })
        
        # 更新代理健康观点
        agent1.update_health_opinion(agent1_reflection["updated_opinion"], agent1_reflection["reason"])
        agent2.update_health_opinion(agent2_reflection["updated_opinion"], agent2_reflection["reason"])
        
        # 将对话整合为情节记忆
        agent1.memory_system.consolidate_dialogue_to_episodic([agent1.name, agent2.name])
        agent2.memory_system.consolidate_dialogue_to_episodic([agent1.name, agent2.name])
        
        # 完成对话
        self.complete_dialogue(dialogue_id)
        
        return dialogue_id

# 创建全局对话管理器实例
dialogue_manager = DialogueManager()

def get_dialogue_manager() -> DialogueManager:
    """获取对话管理器实例
    
    Returns:
        对话管理器实例
    """
    return dialogue_manager