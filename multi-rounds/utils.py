import random
import os
import json
import networkx as nx
import numpy as np
import pandas as pd  # 添加pandas导入
from prompt import *

# 从xlsx加载真实用户数据
def load_real_users(file_path):
    '''从xlsx或csv文件加载真实用户数据，如果失败则直接报错'''
    try:
        # 检查文件类型
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        
        if df.empty:
            raise ValueError("用户数据文件为空")
            
        users = []
        
        for _, row in df.iterrows():
            # 提取大五人格特质
            traits = {
                "开放性": row["开放性"] if pd.notna(row["开放性"]) else "中",
                "尽责性": row["尽责性"] if pd.notna(row["尽责性"]) else "中",
                "外向性": row["外向性"] if pd.notna(row["外向性"]) else "中",
                "宜人性": row["宜人性"] if pd.notna(row["宜人性"]) else "中",
                "神经质": row["神经质"] if pd.notna(row["神经质"]) else "中"
            }
            
            # 创建用户数据字典
            user = {
                "id": str(row["用户id"]) if pd.notna(row["用户id"]) else str(len(users)),
                "name": row["用户名"] if pd.notna(row["用户名"]) else f"未命名用户{len(users)}",
                "description": row["自我描述"] if pd.notna(row["自我描述"]) else "",
                "education": row["教育背景"] if pd.notna(row["教育背景"]) else "未知",
                "traits": traits,
                "health_opinion": row["健康观点"] if pd.notna(row["健康观点"]) else ""
            }
            users.append(user)
        
        if not users:
            raise ValueError("没有从文件中读取到有效用户数据")
            
        print(f"成功加载真实用户数据")
        return users
    except Exception as e:
        print(f"加载真实用户数据失败: {e}")
        raise RuntimeError(f"必须提供有效的用户数据文件且包含足够的用户数据。错误: {e}")

def update_day(agent):
    '''更新代理人的健康状态'''
    # 如果代理人相信谣言，则变为感染状态
    if agent.beliefs[-1] == 1 and agent.health_condition != "Infected":
        agent.health_condition = "Infected"
        agent.model.susceptible -= 1
        agent.model.infected += 1
        agent.model.daily_new_infected_cases += 1
    
    # 如果代理人不再相信谣言，则变为恢复状态
    elif agent.beliefs[-1] == 0 and agent.health_condition == "Infected":
        agent.health_condition = "Recovered"
        agent.model.infected -= 1
        agent.model.recovered += 1
        agent.model.daily_new_recovered_cases += 1

def clear_cache():
    '''清除缓存文件'''
    cache_dir = ".cache"
    if os.path.exists(cache_dir):
        for file in os.listdir(cache_dir):
            os.remove(os.path.join(cache_dir, file))
        print("缓存已清除")

# 创建社交网络
def create_social_network(agents, connection_probability=0.2):  # 降低默认连接概率
    G = nx.Graph()
    
    # 添加节点
    for agent in agents:
        G.add_node(agent.unique_id)
    
    # 添加边
    for i, agent1 in enumerate(agents):
        # 限制每个代理人的最大连接数
        max_connections = min(20, len(agents)//5)  # 每个代理人最多20个连接
        current_connections = 0
        
        for j, agent2 in enumerate(agents[i+1:], i+1):
            # 如果已达最大连接数，跳过
            if current_connections >= max_connections:
                break
                
            # 基础连接概率
            p = connection_probability
            
            # 根据教育背景调整概率
            if agent1.qualification == agent2.qualification:
                p += 0.1  # 教育背景相同的人更可能互动
            
            # 确保概率在有效范围内
            p = max(0.05, min(0.9, p))
            
            # 随机决定是否添加连接
            if random.random() < p:
                # 添加边，权重表示连接强度
                weight = random.uniform(0.5, 1.0)
                G.add_edge(agent1.unique_id, agent2.unique_id, weight=weight)
                current_connections += 1
    
    return G

# 新增：对话状态类
class DialogueState:
    '''管理对话状态的类'''
    def __init__(self, topic, agent1_id, agent2_id):
        self.topic = topic
        self.agent1_id = agent1_id
        self.agent2_id = agent2_id
        self.turn_count = 0
        self.stance_strength = {
            agent1_id: 0,
            agent2_id: 0
        }
        self.common_ground = 0
        self.belief_shifts = {
            agent1_id: 0,
            agent2_id: 0
        }
        self.stop_reason = None
    
    def update_after_turn(self, agent_id, response_data):
        '''更新对话状态'''
        # 更新轮次计数
        if agent_id == self.agent1_id:
            self.turn_count += 0.5  # 半轮
        
        # 更新立场强度
        if "stance_strength" in response_data:
            self.stance_strength[agent_id] = response_data["stance_strength"]
        elif "internal_thoughts" in response_data:
            # 根据内部想法估计立场强度
            thoughts = response_data["internal_thoughts"].lower()
            if "坚定" in thoughts or "确信" in thoughts:
                self.stance_strength[agent_id] = 1.0
            elif "怀疑" in thoughts or "不确定" in thoughts:
                self.stance_strength[agent_id] = 0.5
            elif "不相信" in thoughts or "反对" in thoughts:
                self.stance_strength[agent_id] = -1.0
        
        # 更新共同点
        if "common_ground" in response_data:
            self.common_ground = response_data["common_ground"]
        
        # 更新信念变化
        if "belief_shift" in response_data:
            self.belief_shifts[agent_id] += response_data["belief_shift"]

# 对话停止条件
def should_stop_dialogue(dialogue_state, response1, response2, max_turns=3, convergence_threshold=0.1):
    '''判断对话是否应该停止'''
    # 检查最大轮次
    if dialogue_state.turn_count >= max_turns:
        dialogue_state.stop_reason = "达到最大轮次"
        return True
    
    # 检查对话收敛 - 获取最近一轮的信念变化
    recent_shift1 = abs(response1.get("belief_shift", 0))
    recent_shift2 = abs(response2.get("belief_shift", 0))
    
    if recent_shift1 < convergence_threshold and recent_shift2 < convergence_threshold:
        dialogue_state.stop_reason = "对话收敛"
        return True
    
    # 检查明确的结束信号
    if "response" in response1 and ("结束" in response1["response"] or "再见" in response1["response"]):
        dialogue_state.stop_reason = "代理人1明确结束"
        return True
    
    if "response" in response2 and ("结束" in response2["response"] or "再见" in response2["response"]):
        dialogue_state.stop_reason = "代理人2明确结束"
        return True
    
    # 默认继续对话
    return False

# 计算最终信念变化
def calculate_final_belief_change(agent, dialogue_state, conversation_history):
    '''计算对话后的最终信念变化'''
    # 获取代理人ID
    agent_id = agent.unique_id
    
    # 基础信念变化来自对话状态
    belief_change = dialogue_state.belief_shifts.get(agent_id, 0)
    
    # 分析对话内容进行调整
    for turn in conversation_history:
        if turn["speaker"] == agent.name:
            content = turn["content"].lower()
            # 检测强烈的信念变化信号
            if "我改变主意了" in content or "你说服了我" in content:
                if agent.beliefs[-1] == 1:  # 如果当前相信
                    belief_change = -1.0  # 强烈的负向变化
                else:  # 如果当前不相信
                    belief_change = 1.0  # 强烈的正向变化
                break
    
    # 根据对话轮次调整变化强度
    if dialogue_state.turn_count < 1:
        belief_change *= 0.5  # 对话太短，影响减半
    
    return belief_change

# 格式化对话历史
def format_dialogue_history(conversation_history):
    '''将对话历史格式化为字符串'''
    if not conversation_history:
        return "（无对话历史）"
    
    formatted = ""
    for turn in conversation_history:
        formatted += f"{turn['speaker']}: {turn['content']}\n"
    
    return formatted

# 获取对话摘要
def get_dialogue_summary(dialogue_content, topic):
    '''获取对话内容的摘要'''
    user_msg = dialogue_summary_prompt.format(
        dialogue_content=dialogue_content,
        topic=topic
    )
    
    msg = [{"role": "user", "content": user_msg}]
    response = get_completion_from_messages(msg, temperature=0.5)
    
    return response

# 新增：从健康观点生成长期记忆
def create_memory_from_health_opinion(health_opinion, name):
    '''从健康观点创建初始长期记忆'''
    if not health_opinion:
        return ""
    
    # 创建一个基于健康观点的长期记忆
    memory = f"我是{name}，我对健康的看法是：{health_opinion}\n\n"
    memory += f"这是我基于个人经验和知识形成的观点，我相信这些健康理念。"
    
    return memory