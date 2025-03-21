import os
import yaml
import argparse
from typing import Dict, Any, Optional

class ConfigLoader:
    """配置加载器: 负责从YAML文件加载配置并提供统一的访问接口"""
    
    def __init__(self, config_dir="config"):
        """初始化配置加载器
        
        Args:
            config_dir: 配置文件目录，默认为'config'
        """
        self.config_dir = config_dir
        self.config_data = {}
        self.load_default_config()
    
    def load_default_config(self):
        """加载默认配置文件"""
        # 加载default.yaml
        default_config_path = os.path.join(self.config_dir, "default.yaml")
        if os.path.exists(default_config_path):
            with open(default_config_path, 'r', encoding='utf-8') as f:
                self.config_data.update(yaml.safe_load(f) or {})
        
        # 加载其他配置文件
        for config_file in ["simulation.yaml", "agent.yaml", "dialogue.yaml"]:
            config_path = os.path.join(self.config_dir, config_file)
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    # 使用文件名作为配置节点的键
                    section_name = config_file.split(".")[0]
                    self.config_data[section_name] = yaml.safe_load(f) or {}
        
        # 加载主题配置
        topics_dir = os.path.join(self.config_dir, "topics")
        if os.path.exists(topics_dir):
            self.config_data["topics"] = {}
            for topic_file in os.listdir(topics_dir):
                if topic_file.endswith(".yaml"):
                    topic_path = os.path.join(topics_dir, topic_file)
                    topic_name = topic_file.split(".")[0]
                    with open(topic_path, 'r', encoding='utf-8') as f:
                        self.config_data["topics"][topic_name] = yaml.safe_load(f) or {}
    
    def load_custom_config(self, config_path):
        """加载自定义配置文件并覆盖默认配置
        
        Args:
            config_path: 自定义配置文件路径
        """
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                custom_config = yaml.safe_load(f) or {}
                # 递归更新配置
                self._update_dict_recursive(self.config_data, custom_config)
    
    def _update_dict_recursive(self, d, u):
        """递归更新字典
        
        Args:
            d: 目标字典
            u: 更新源字典
        """
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_dict_recursive(d[k], v)
            else:
                d[k] = v
    
    def get(self, key, default=None):
        """获取配置值
        
        Args:
            key: 配置键，支持点号分隔的路径，如'simulation.days'
            default: 如果键不存在，返回的默认值
            
        Returns:
            配置值或默认值
        """
        keys = key.split('.')
        value = self.config_data
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def update_from_args(self, args):
        """从命令行参数更新配置
        
        Args:
            args: 解析后的命令行参数
        """
        # 将命令行参数转换为字典
        args_dict = vars(args)
        # 更新配置
        for k, v in args_dict.items():
            # 如果参数具有默认值且未被用户覆盖，则跳过
            if v is None:
                continue
            
            # 根据参数名称确定配置节点
            if k.startswith('no_'):
                section = 'simulation'
                key = k[3:]  # 移除'no_'前缀
                if key == 'days':
                    key = 'days'
                elif key == 'init_healthy':
                    key = 'initial_healthy'
                elif key == 'init_infect':
                    key = 'initial_infected'
                elif key == 'of_runs':
                    key = 'runs'
            elif k in ['contact_rate', 'name', 'offset']:
                section = 'simulation'
                key = k
            elif k in ['max_dialogue_turns', 'dialogue_convergence', 'save_dialogues']:
                section = 'dialogue'
                key = k
            elif k == 'save_behaviors':
                section = 'agent'
                key = 'behavior_logging'
            elif k == 'user_data_file':
                section = 'paths'
                key = 'user_data_file'
            else:
                # 其他参数直接在根级别设置
                self.config_data[k] = v
                continue
            
            # 确保节点存在
            if section not in self.config_data:
                self.config_data[section] = {}
            
            # 对于paths节点特殊处理
            if section == 'paths':
                if 'paths' not in self.config_data:
                    self.config_data['paths'] = {}
                self.config_data['paths'][key] = v
            else:
                # 更新配置
                self.config_data[section][key] = v
    
    def get_all(self):
        """获取所有配置数据
        
        Returns:
            包含所有配置的字典
        """
        return self.config_data
    
    def print_config(self):
        """打印当前配置内容（用于调试）"""
        import json
        print(json.dumps(self.config_data, indent=2, ensure_ascii=False))

# 创建全局配置实例
config = ConfigLoader()

def get_config(key=None, default=None):
    """获取配置的便捷函数
    
    Args:
        key: 配置键，支持点号分隔的路径，如'simulation.days'
        default: 如果键不存在，返回的默认值
            
    Returns:
        如果key为None，返回整个配置字典；否则返回指定的配置值或默认值
    """
    if key is None:
        return config.get_all()
    return config.get(key, default)

def load_config(config_path=None, args=None):
    """加载配置的便捷函数
    
    Args:
        config_path: 自定义配置文件路径
        args: 解析后的命令行参数
    """
    if config_path:
        config.load_custom_config(config_path)
    if args:
        config.update_from_args(args)
    
    return config
