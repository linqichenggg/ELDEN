import os
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from datetime import datetime
import random

def calculate_rumor_ratio(self):
    """计算当前网络中谣言的比例"""
    if not self.messages_in_network:
        return 0
        
    # 计算最近一个时间步骤内的消息
    recent_messages = [msg for msg in self.messages_in_network 
                      if msg["created_at"] >= self.current_time_step - 1]
    
    if not recent_messages:
        return 0
        
    rumor_count = sum(1 for msg in recent_messages if msg["type"] == "rumor")
    return rumor_count / len(recent_messages)

def check_termination_condition(self):
    """检查是否满足终止条件"""
    # 例如：如果没有新消息传播，或者谣言比例低于阈值
    recent_messages = [msg for msg in self.messages_in_network 
                      if msg["created_at"] == self.current_time_step]
    
    return len(recent_messages) == 0

def activate_debunking_bot(self):
    """激活辟谣机器人"""
    self.debunking_bot_active = True
    self.log("辟谣机器人已激活")
    
    # 获取所有正在传播的谣言
    active_rumors = set()
    for msg in self.messages_in_network:
        if msg["type"] == "rumor" and msg["created_at"] >= self.current_time_step - 2:
            active_rumors.add(msg["original_id"])
    
    # 为每个谣言发送辟谣信息
    for rumor_id in active_rumors:
        # 查找原始谣言内容
        rumor_content = next((msg["content"] for msg in self.information_database 
                             if msg["id"] == rumor_id), "")
        
        if not rumor_content:
            continue
            
        # 创建辟谣信息
        debunk_content = f"辟谣通知：您可能收到的信息'{rumor_content}'是虚假的。请勿相信和传播。专家建议：健康信息请遵循医生建议，不要轻信网络传闻。"
        
        debunk_message = {
            "id": f"debunk_{rumor_id}_{int(time.time())}",
            "content": debunk_content,
            "type": "debunk",
            "original_id": f"debunk_{rumor_id}",
            "sender": "debunk_bot",
            "created_at": self.current_time_step,
            "path": ["debunk_bot"],
            "verified_by": [],
            "target_rumor": rumor_id
        }
        
        # 向所有用户发送辟谣信息
        for user_id in self.users:
            # 只发送给最近收到谣言的用户
            received_rumors = [msg for msg in self.messages_in_network 
                              if msg["id"] in self.users[user_id]["received_messages"]
                              and msg["original_id"] == rumor_id
                              and msg["created_at"] >= self.current_time_step - 2]
            
            if received_rumors:
                # 用户接收辟谣信息
                self.users[user_id]["received_messages"].append(debunk_message["id"])
                
                # 找到原始谣言
                original_rumor = next((r for r in self.information_database if r["id"] == rumor_id), None)
                
                if original_rumor and self.llm:
                    # 调用LLM模拟用户对辟谣的反应
                    reaction = self.call_llm_for_debunk_reaction(user_id, debunk_message, original_rumor)
                    
                    # 处理用户反应
                    if reaction["changed_mind"]:
                        # 用户改变想法
                        # 如果该用户转发过这个谣言，根据反应决定下一步行动
                        forwarded_rumor_ids = [
                            msg["original_id"] for msg in self.messages_in_network
                            if msg["id"] in self.users[user_id]["forwarded_messages"] and msg["original_id"] == rumor_id
                        ]
                        
                        if forwarded_rumor_ids and reaction["action"] == "删除":
                            # 模拟删除行为（从转发列表中移除）
                            for msg_id in list(self.users[user_id]["forwarded_messages"]):
                                msg = next((m for m in self.messages_in_network if m["id"] == msg_id), None)
                                if msg and msg["original_id"] == rumor_id:
                                    self.users[user_id]["forwarded_messages"].remove(msg_id)
                
        # 添加辟谣消息到网络
        self.messages_in_network.append(debunk_message)

def save_simulation_results(self):
    """保存模拟结果"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(self.save_dir, f"simulation_{timestamp}")
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 保存用户数据
    users_df = pd.DataFrame.from_dict(self.users, orient='index')
    users_df['received_count'] = users_df['received_messages'].apply(len)
    users_df['forwarded_count'] = users_df['forwarded_messages'].apply(len)
    users_df['verified_count'] = users_df['verified_messages'].apply(len)
    
    # 移除大列表以便保存
    users_df = users_df.drop(columns=['received_messages', 'forwarded_messages', 'verified_messages'])
    users_df.to_csv(os.path.join(results_dir, 'users.csv'))
    
    # 保存消息数据
    messages_df = pd.DataFrame(self.messages_in_network)
    messages_df['path_length'] = messages_df['path'].apply(len)
    messages_df['verified_count'] = messages_df['verified_by'].apply(len)
    
    # 临时处理路径和验证者列表以便保存
    messages_df['path_str'] = messages_df['path'].apply(lambda x: ','.join(x))
    messages_df['verified_by_str'] = messages_df['verified_by'].apply(lambda x: ','.join(x) if x else '')
    
    messages_df = messages_df.drop(columns=['path', 'verified_by'])
    messages_df.to_csv(os.path.join(results_dir, 'messages.csv'))
    
    # 保存网络数据 - 创建一个只包含简单数据类型的网络
    simplified_network = nx.DiGraph()
    
    # 复制节点但只包含简单数据类型的属性
    for node, attrs in self.social_network.nodes(data=True):
        clean_attrs = {}
        for key, value in attrs.items():
            # 只保留简单数据类型
            if isinstance(value, (str, int, float, bool)) and value is not None:
                clean_attrs[key] = value
            elif value is None:
                clean_attrs[key] = "None"  # 将None转换为字符串
        
        simplified_network.add_node(node, **clean_attrs)
    
    # 复制边
    for u, v, attrs in self.social_network.edges(data=True):
        clean_attrs = {}
        for key, value in attrs.items():
            if isinstance(value, (str, int, float, bool)) and value is not None:
                clean_attrs[key] = value
            elif value is None:
                clean_attrs[key] = "None"
        
        simplified_network.add_edge(u, v, **clean_attrs)
    
    # 保存简化后的网络
    nx.write_gexf(simplified_network, os.path.join(results_dir, 'social_network.gexf'))
    
    # 保存模拟配置
    config = {
        'initial_users': self.initial_users,
        'max_time_steps': self.max_time_steps,
        'actual_time_steps': self.current_time_step,
        'debunking_threshold': self.debunking_threshold,
        'debunking_bot_active': self.debunking_bot_active,
        'use_llm': self.use_llm,
        'llm_model': self.model_name if self.use_llm else None,
        'timestamp': timestamp
    }
    
    with open(os.path.join(results_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # 分析结果并保存报告
    self.analyze_results(results_dir)
    
    self.log(f"模拟结果已保存到: {results_dir}")
    return results_dir

def analyze_results(self, results_dir):
    """分析模拟结果并生成报告"""
    # 加载保存的数据
    users_df = pd.read_csv(os.path.join(results_dir, 'users.csv'), index_col=0)
    messages_df = pd.read_csv(os.path.join(results_dir, 'messages.csv'))
    
    # 1. 谣言存活分析
    rumor_survival = messages_df[messages_df['type'] == 'rumor'].groupby('original_id')['path_length'].max()
    
    # 2. 用户传播行为分析
    # 找出传播谣言的用户
    rumor_messages = messages_df[messages_df['type'] == 'rumor']
    rumor_forwarders = []
    
    for user_id, user in self.users.items():
        forwarded_rumors = [
            msg_id for msg_id in user['forwarded_messages']
            if any(msg['id'] == msg_id and msg['type'] == 'rumor' for msg in self.messages_in_network)
        ]
        
        if forwarded_rumors:
            rumor_forwarders.append({
                'id': user_id,
                'health_knowledge': user['health_knowledge'],
                'transmission_tendency': user['transmission_tendency'],
                'rumor_sensitivity': user['rumor_sensitivity'],
                'social_circle_size': user['social_circle_size'],
                'rumor_forwards': len(forwarded_rumors)
            })
    
    rumor_forwarders_df = pd.DataFrame(rumor_forwarders)
    
    # 找出top 10%的谣言传播者
    if not rumor_forwarders_df.empty and len(rumor_forwarders_df) >= 10:
        top_forwarders = rumor_forwarders_df.nlargest(int(len(rumor_forwarders_df) * 0.1), 'rumor_forwards')
        top_forwarders.to_csv(os.path.join(results_dir, 'top_rumor_forwarders.csv'))
    
    # 3. 谣言传播效率分析
    rumor_efficiency = messages_df[messages_df['type'] == 'rumor'].groupby('original_id').agg({
        'path_length': ['min', 'max', 'mean', 'count']
    })
    
    # 4. 验证行为分析
    verification_rates = messages_df.groupby('type')['verified_count'].mean()
    
    # 将分析结果保存为报告
    with open(os.path.join(results_dir, 'analysis_report.txt'), 'w') as f:
        f.write("=== 模拟分析报告 ===\n\n")
        
        f.write("1. 谣言存活轮次:\n")
        f.write(str(rumor_survival) + "\n\n")
        
        f.write("2. 谣言传播效率:\n")
        f.write(str(rumor_efficiency) + "\n\n")
        
        f.write("3. 信息验证率:\n")
        f.write(str(verification_rates) + "\n\n")
        
        if not rumor_forwarders_df.empty:
            f.write("4. 谣言传播者特征总结:\n")
            f.write(f"平均健康知识水平: {rumor_forwarders_df['health_knowledge'].mean():.2f}\n")
            f.write(f"平均传播倾向: {rumor_forwarders_df['transmission_tendency'].mean():.2f}\n")
            f.write(f"平均谣言敏感度: {rumor_forwarders_df['rumor_sensitivity'].mean():.2f}\n")
            f.write(f"平均社交圈大小: {rumor_forwarders_df['social_circle_size'].mean():.2f}\n\n")
        
        if 'top_forwarders' in locals():
            f.write("5. Top 10% 谣言传播者特征:\n")
            f.write(f"平均健康知识水平: {top_forwarders['health_knowledge'].mean():.2f}\n")
            f.write(f"平均传播倾向: {top_forwarders['transmission_tendency'].mean():.2f}\n")
            f.write(f"平均谣言敏感度: {top_forwarders['rumor_sensitivity'].mean():.2f}\n")
            f.write(f"平均社交圈大小: {top_forwarders['social_circle_size'].mean():.2f}\n")
    
    # 创建可视化
    self.create_visualizations(results_dir, users_df, messages_df)

def visualize_network(self, results_dir):
    """可视化社交网络"""
    plt.figure(figsize=(12, 12))
    
    # 创建网络布局
    pos = nx.spring_layout(self.social_network, seed=42)
    
    # 按健康知识水平着色节点
    knowledge_values = [data['health_knowledge'] for _, data in self.social_network.nodes(data=True)]
    
    # 按是否传播谣言调整节点大小
    node_sizes = []
    for node, data in self.social_network.nodes(data=True):
        forwarded_rumors = 0
        for msg_id in self.users[node]['forwarded_messages']:
            msg = next((m for m in self.messages_in_network if m["id"] == msg_id), None)
            if msg and msg["type"] == "rumor":
                forwarded_rumors += 1
        
        # 基础大小 + 谣言传播次数
        node_sizes.append(50 + forwarded_rumors * 20)
    
    # 绘制网络
    nodes = nx.draw_networkx_nodes(
        self.social_network, 
        pos, 
        node_color=knowledge_values,
        node_size=node_sizes,
        cmap=plt.cm.viridis,
        alpha=0.8
    )
    
    edges = nx.draw_networkx_edges(
        self.social_network,
        pos,
        alpha=0.5,
        arrows=True
    )
    
    # 添加颜色条和标题
    plt.colorbar(nodes, label="health knowledge level")
    plt.title("Social network structure")
    
    # 关闭坐标轴
    plt.axis('off')
    
    # 保存图形
    plt.savefig(os.path.join(results_dir, 'social_network.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_visualizations(self, results_dir, users_df, messages_df):
    """创建各种可视化图表"""
    # 1. 可视化社交网络
    self.visualize_network(results_dir)
    
    # 2. 可视化谣言传播情况
    rumor_counts = messages_df[messages_df['type'] == 'rumor'].groupby('created_at').size()
    official_counts = messages_df[messages_df['type'] == 'official'].groupby('created_at').size()
    
    plt.figure(figsize=(10, 6))
    plt.plot(rumor_counts.index, rumor_counts.values, 'r-', label='rumor')
    plt.plot(official_counts.index, official_counts.values, 'g-', label='official')
    
    if self.debunking_bot_active:
        debunk_time = min(m['created_at'] for m in self.messages_in_network if m['type'] == 'debunk')
        plt.axvline(x=debunk_time, color='b', linestyle='--', label='robot activation')
    
    plt.xlabel('time step')
    plt.ylabel('message count')
    plt.title('rumor and official information propagation over time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(results_dir, 'message_propagation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 可视化用户特征分布
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    sns.histplot(users_df['health_knowledge'], kde=True)
    plt.title('health knowledge level distribution')
    
    plt.subplot(2, 2, 2)
    sns.histplot(users_df['transmission_tendency'], kde=True)
    plt.title('transmission tendency distribution')
    
    plt.subplot(2, 2, 3)
    sns.histplot(users_df['rumor_sensitivity'], kde=True)
    plt.title('rumor sensitivity distribution')
    
    plt.subplot(2, 2, 4)
    sns.histplot(users_df['social_circle_size'], kde=True)
    plt.title('social circle size distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'user_feature_distributions.png'), dpi=300)
    plt.close()
    
    # 4. 谣言传播者vs非传播者特征对比
    rumor_forwarder_ids = []
    for user_id, user in self.users.items():
        for msg_id in user['forwarded_messages']:
            msg = next((m for m in self.messages_in_network if m["id"] == msg_id), None)
            if msg and msg["type"] == "rumor":
                rumor_forwarder_ids.append(user_id)
                break
    
    users_df['is_rumor_forwarder'] = users_df.index.isin(rumor_forwarder_ids)
    
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    sns.boxplot(x='is_rumor_forwarder', y='health_knowledge', data=users_df)
    plt.title('health knowledge level vs rumor propagation')
    plt.xlabel('whether the user is a rumor forwarder')
    
    plt.subplot(2, 2, 2)
    sns.boxplot(x='is_rumor_forwarder', y='transmission_tendency', data=users_df)
    plt.title('transmission tendency vs rumor propagation')
    plt.xlabel('whether the user is a rumor forwarder')
    
    plt.subplot(2, 2, 3)
    sns.boxplot(x='is_rumor_forwarder', y='rumor_sensitivity', data=users_df)
    plt.title('rumor sensitivity vs rumor propagation')
    plt.xlabel('whether the user is a rumor forwarder')
    
    plt.subplot(2, 2, 4)
    sns.boxplot(x='is_rumor_forwarder', y='social_circle_size', data=users_df)
    plt.title('social circle size vs rumor propagation')
    plt.xlabel('whether the user is a rumor forwarder')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'rumor_forwarder_comparison.png'), dpi=300)
    plt.close()