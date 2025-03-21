from typing import Dict, List, Any, Optional, Tuple
import time
import random

from config_loader import get_config
from utils.api_client import get_api_client
from utils.prompt_templates import get_prompt_manager

class BeliefSystem:
    """信念系统，管理代理的信念和更新机制"""
    
    def __init__(self, agent_id: str, agent_name: str, agent_traits: Dict[str, float], agent_age: int):
        """初始化信念系统
        
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
        
        # API客户端
        self.api_client = get_api_client()
        
        # 提示模板管理器
        self.prompt_manager = get_prompt_manager()
        
        # 基本信念
        self.beliefs = {}
        
        # 信任倾向（受外向性和宜人性影响）
        self.trust_tendency = self._calculate_trust_tendency()
        
        # 信息接受倾向（受开放性影响）
        self.info_acceptance = self._calculate_info_acceptance()
        
        # 稳定性（受神经质影响）
        self.stability = self._calculate_stability()
    
    def _calculate_trust_tendency(self) -> float:
        """计算信任倾向
        
        Returns:
            信任倾向（0.0-1.0）
        """
        # 信任倾向受外向性和宜人性影响
        extraversion = self.agent_traits.get("外向性", 0.5)
        agreeableness = self.agent_traits.get("宜人性", 0.5)
        
        # 加权平均，宜人性影响更大
        return 0.3 * extraversion + 0.7 * agreeableness
    
    def _calculate_info_acceptance(self) -> float:
        """计算信息接受倾向
        
        Returns:
            信息接受倾向（0.0-1.0）
        """
        # 信息接受倾向受开放性影响
        openness = self.agent_traits.get("开放性", 0.5)
        return openness
    
    def _calculate_stability(self) -> float:
        """计算稳定性
        
        Returns:
            稳定性（0.0-1.0）
        """
        # 稳定性受神经质影响（负相关）
        neuroticism = self.agent_traits.get("神经质", 0.5)
        return 1.0 - neuroticism
    
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
    
    def set_belief(self, category: str, content: str, confidence: float = 0.5):
        """设置信念
        
        Args:
            category: 信念类别
            content: 信念内容
            confidence: 确信度（0.0-1.0）
        """
        self.beliefs[category] = {
            "content": content,
            "confidence": confidence,
            "last_updated": time.time()
        }
    
    def get_belief(self, category: str) -> Optional[Dict[str, Any]]:
        """获取信念
        
        Args:
            category: 信念类别
            
        Returns:
            信念字典，如果不存在则返回None
        """
        return self.beliefs.get(category)
    
    def get_belief_content(self, category: str, default: str = "") -> str:
        """获取信念内容
        
        Args:
            category: 信念类别
            default: 默认值
            
        Returns:
            信念内容，如果不存在则返回默认值
        """
        belief = self.get_belief(category)
        if belief:
            return belief["content"]
        return default
    
    def update_belief(self, category: str, new_information: str) -> Tuple[str, float, str]:
        """更新信念
        
        Args:
            category: 信念类别
            new_information: 新信息
            
        Returns:
            (更新后的信念内容, 变化程度, 原因分析)
        """
        current_belief = self.get_belief_content(category, "我对此没有特别的看法。")
        
        # 调用LLM进行信念更新
        prompt = self.prompt_manager.format_prompt("belief_update",
            agent_name=self.agent_name,
            agent_age=self.agent_age,
            agent_traits=self._format_traits(),
            current_belief=current_belief,
            new_information=new_information
        )
        
        success, response = self.api_client.generate(prompt)
        if success:
            response_text = response["response"]
            
            # 解析响应
            updated_opinion = current_belief  # 默认不变
            change_degree = 0
            reason = "信息不足以改变观点"
            
            for line in response_text.split("\n"):
                if line.startswith("更新后的观点:"):
                    updated_opinion = line.replace("更新后的观点:", "").strip()
                elif line.startswith("变化程度:"):
                    try:
                        change_str = line.replace("变化程度:", "").strip()
                        change_degree = float(change_str) / 10.0  # 归一化到0-1
                    except ValueError:
                        change_degree = 0
                elif line.startswith("原因分析:"):
                    reason = line.replace("原因分析:", "").strip()
            
            # 根据代理特质调整变化程度
            # 信息接受倾向高的代理变化可能更大
            adjusted_change = change_degree * self.info_acceptance
            
            # 稳定性高的代理变化可能更小
            adjusted_change *= (2 - self.stability)
            
            # 根据调整后的变化程度决定是否更新信念
            if adjusted_change > 0.2:  # 阈值
                confidence = min(0.5 + adjusted_change / 2, 0.9)
                self.set_belief(category, updated_opinion, confidence)
            
            return updated_opinion, adjusted_change, reason
        
        # 如果生成失败，则保持不变
        return current_belief, 0, "无法处理新信息"
    
    def evaluate_info_credibility(self, information: str, source: str) -> float:
        """评估信息可信度
        
        Args:
            information: 信息内容
            source: 信息来源
            
        Returns:
            信息可信度（0.0-1.0）
        """
        # 简单实现：结合信任倾向和随机因素
        base_credibility = 0.5
        
        # 来源影响
        if source == "权威专家":
            base_credibility += 0.3
        elif source == "亲密朋友":
            base_credibility += 0.2 * self.trust_tendency
        elif source == "新闻媒体":
            base_credibility += 0.1
        elif source == "社交媒体":
            base_credibility -= 0.1
        elif source == "未知来源":
            base_credibility -= 0.2
        
        # 随机波动
        random_factor = random.uniform(-0.1, 0.1)
        
        # 最终可信度
        credibility = base_credibility + random_factor
        
        # 确保在0-1范围内
        return max(0.0, min(1.0, credibility))
    
    def process_new_information(self, category: str, information: str, source: str) -> Tuple[str, float, str]:
        """处理新信息
        
        Args:
            category: 信念类别
            information: 信息内容
            source: 信息来源
            
        Returns:
            (更新后的信念内容, 变化程度, 原因分析)
        """
        # 评估信息可信度
        credibility = self.evaluate_info_credibility(information, source)
        
        # 如果可信度太低，直接拒绝
        if credibility < 0.3:
            current_belief = self.get_belief_content(category, "我对此没有特别的看法。")
            return current_belief, 0, f"来自{source}的信息可信度不足"
        
        # 更新信念
        return self.update_belief(category, information)