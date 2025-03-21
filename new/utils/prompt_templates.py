from typing import Dict, Any, Optional, List
import json
import os
from string import Template

from config_loader import get_config

class PromptTemplate:
    """提示模板类，用于管理和格式化提示模板"""
    
    def __init__(self, template_str: str):
        """初始化提示模板
        
        Args:
            template_str: 模板字符串，使用$变量名或${变量名}作为占位符
        """
        self.template = Template(template_str)
    
    def format(self, **kwargs) -> str:
        """格式化模板
        
        Args:
            **kwargs: 用于替换模板中占位符的变量
            
        Returns:
            格式化后的字符串
        """
        return self.template.safe_substitute(**kwargs)


class PromptManager:
    """提示模板管理器，用于管理所有提示模板"""
    
    def __init__(self):
        """初始化提示模板管理器"""
        self.templates = {}
        self.load_default_templates()
    
    def load_default_templates(self):
        """加载默认提示模板"""
        # 对话系统模板
        self._add_template("agent_interaction", """
你是一个名叫${agent_name}的${agent_age}岁老年人，你的性格特点是：${agent_traits}。
你正在与${other_name}讨论关于健康的话题。

${other_name}告诉你: "${message}"

请基于你的个性回应这条消息，考虑你的个性特点、背景，以及你对健康话题的看法。在讨论中尝试表达你自己的观点。

你对健康话题的观点是: ${health_opinion}

使用第一人称回复，不要说"作为[你的名字]"或类似的话。不需重复问题或解释你是谁，直接以对话形式回应即可。
""")

        self._add_template("agent_reflection", """
你是一个名叫${agent_name}的${agent_age}岁老年人，你的性格特点是：${agent_traits}。

你刚刚与${other_name}进行了一段关于健康的对话:

对话历史:
${dialogue_history}

基于这次对话，请反思并更新你对健康相关话题的看法。考虑对方的观点如何影响了你的想法。

你之前对健康话题的观点是: ${previous_opinion}

请提供:
1. 你更新后的健康观点（如果有变化）
2. 简短解释这种变化（或保持不变）的原因

请使用如下格式回答:
更新观点: [您更新后的健康观点]
原因: [解释原因]
""")

        self._add_template("memory_consolidation", """
你是一个名叫${agent_name}的${agent_age}岁老年人，你的性格特点是：${agent_traits}。

请根据以下关于你的记忆片段，创建一个简洁的记忆摘要:

记忆片段:
${memory_fragments}

请生成一个不超过100字的摘要，捕捉这些记忆的核心内容和情感。摘要应该反映你的个性特点，并专注于对你来说最重要的细节。
""")

        # 信念系统模板
        self._add_template("belief_update", """
你是一个名叫${agent_name}的${agent_age}岁老年人，你的性格特点是：${agent_traits}。

你目前对健康话题的看法是:
${current_belief}

你刚刚接触到以下信息:
${new_information}

考虑到你的个性特点，请分析这个新信息会如何影响你的健康观点。特别考虑:
1. 你对新信息的接受程度
2. 这个信息如何与你现有的观点相符或冲突
3. 你的性格如何影响你处理新信息的方式

请提供:
更新后的观点: [如有变化，描述你新的健康观点]
变化程度: [1-10的数字，1表示几乎没变化，10表示显著变化]
原因分析: [简短解释为什么会有这种变化或保持不变]
""")

        # 健康谣言模板
        self._add_template("rumor_analysis", """
你是一个AI助手，专门分析健康谣言的传播模式。

请分析以下健康相关信息:
"${rumor_content}"

根据数据分析，这条信息:
1. 主题类别: [例如: 营养、药物、疾病预防等]
2. 可信度评估: [高/中/低] - 基于科学证据
3. 误导性元素: [列出任何误导性元素]
4. 可能传播原因: [心理或社会因素分析]
5. 校正建议: [如何提供准确信息]

请提供一个客观的、基于证据的分析。
""")
    
    def _add_template(self, name: str, template_str: str):
        """添加模板
        
        Args:
            name: 模板名称
            template_str: 模板字符串
        """
        self.templates[name] = PromptTemplate(template_str)
    
    def add_template(self, name: str, template_str: str):
        """添加或更新模板
        
        Args:
            name: 模板名称
            template_str: 模板字符串
        """
        self._add_template(name, template_str)
    
    def add_templates_from_file(self, file_path: str):
        """从JSON文件加载模板
        
        Args:
            file_path: JSON文件路径
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                templates = json.load(f)
                for name, template_str in templates.items():
                    self.add_template(name, template_str)
        except Exception as e:
            print(f"加载模板文件失败: {e}")
    
    def add_templates_from_config(self):
        """从配置加载模板"""
        templates = get_config("templates", {})
        for name, template_str in templates.items():
            self.add_template(name, template_str)
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """获取指定名称的模板
        
        Args:
            name: 模板名称
            
        Returns:
            模板对象，如果不存在则返回None
        """
        return self.templates.get(name)
    
    def format_prompt(self, name: str, **kwargs) -> Optional[str]:
        """格式化指定名称的模板
        
        Args:
            name: 模板名称
            **kwargs: 用于替换模板中占位符的变量
            
        Returns:
            格式化后的字符串，如果模板不存在则返回None
        """
        template = self.get_template(name)
        if template:
            return template.format(**kwargs)
        return None
    
    def list_templates(self) -> List[str]:
        """列出所有可用的模板名称
        
        Returns:
            模板名称列表
        """
        return list(self.templates.keys())
    
    def save_templates(self, file_path: str):
        """将所有模板保存到JSON文件
        
        Args:
            file_path: 保存路径
        """
        try:
            templates_dict = {
                name: self.templates[name].template.template 
                for name in self.templates
            }
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(templates_dict, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存模板文件失败: {e}")

# 创建全局提示模板管理器实例
prompt_manager = PromptManager()

def get_prompt_manager() -> PromptManager:
    """获取提示模板管理器实例
    
    Returns:
        提示模板管理器实例
    """
    return prompt_manager