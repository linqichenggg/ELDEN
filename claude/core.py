import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random
import json
import time
import os
from datetime import datetime
import seaborn as sns
from tqdm import tqdm
from zhipuai import ZhipuAI

class RumorSimulation:
    def __init__(self, api_key=None, model="glm-3-turbo", initial_users=10, 
                 max_time_steps=10, save_dir="simulation_results"):
        # 基本参数设置
        self.model_name = model
        self.api_key = api_key
        self.initial_users = initial_users
        self.max_time_steps = max_time_steps
        self.save_dir = save_dir
        
        # 数据结构初始化
        self.users = {}
        self.social_network = nx.DiGraph()
        self.information_database = []
        self.messages_in_network = []
        self.knowledge_questions = []
        
        # 模拟状态
        self.current_time_step = 0
        self.debunking_bot_active = False
        self.debunking_threshold = 0.3  # 30%虚假信息激活辟谣机器人
        
        # 确保存储目录存在
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # 初始化LLM客户端
        self.llm = None
        self.llm_api_key = api_key
        self.use_llm = api_key is not None  # 添加标志控制是否使用LLM
        
        # 初始化日志
        self.log_file = os.path.join(save_dir, f"simulation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"模拟开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"初始用户数: {initial_users}\n")
            f.write(f"最大时间步长: {max_time_steps}\n\n")
            f.write(f"是否使用LLM: {'是' if self.use_llm else '否'}\n\n")
    
    def log(self, message, level="INFO"):
        """记录日志"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] [{level}] {message}"
        
        print(log_message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + "\n")
    
    def setup_knowledge_questions(self):
        """设置健康知识测试问题"""
        self.log("设置健康知识测试问题...")
        
        # 健康知识问题列表
        self.knowledge_questions = [
            {
                "question": "每天喝8杯水是否对所有人都必要？",
                "options": ["是", "否"],
                "correct": "否",
                "correct_answer": "否"  # 添加两个键名以保持兼容性
            },
            {
                "question": "感冒时是否应该立即服用抗生素？",
                "options": ["是", "否"],
                "correct": "否",
                "correct_answer": "否"
            },
            {
                "question": "高血压患者是否应该完全避免盐分摄入？",
                "options": ["是", "否"],
                "correct": "否",
                "correct_answer": "否"
            },
            {
                "question": "糖尿病患者是否应该完全避免水果？",
                "options": ["是", "否"],
                "correct": "否",
                "correct_answer": "否"
            },
            {
                "question": "每天步行10000步是否是健康标准？",
                "options": ["是", "否"],
                "correct": "否",
                "correct_answer": "否"
            },
            {
                "question": "睡眠时间是否每个人都必须是8小时？",
                "options": ["是", "否"],
                "correct": "否",
                "correct_answer": "否"
            },
            {
                "question": "维生素补充剂是否对所有人都必要？",
                "options": ["是", "否"],
                "correct": "否",
                "correct_answer": "否"
            },
            {
                "question": "关节疼痛是否总是关节炎的信号？",
                "options": ["是", "否"],
                "correct": "否",
                "correct_answer": "否"
            },
            {
                "question": "老年人是否应该避免所有形式的运动？",
                "options": ["是", "否"],
                "correct": "否",
                "correct_answer": "否"
            },
            {
                "question": "饭后立即游泳是否会导致抽筋？",
                "options": ["是", "否"],
                "correct": "是",
                "correct_answer": "是"
            }
        ]
        
        # 随机打乱问题顺序，增加随机性
        random.shuffle(self.knowledge_questions)
        
        # 随机选择一部分问题反转答案，增加随机性
        for q in random.sample(self.knowledge_questions, len(self.knowledge_questions) // 3):
            if q["correct"] == "是":
                q["correct"] = "否"
                q["correct_answer"] = "否"
            else:
                q["correct"] = "是"
                q["correct_answer"] = "是"
            
            # 确保选项顺序与正确答案一致
            if q["correct"] not in q["options"]:
                q["options"] = ["是", "否"] if q["correct"] == "是" else ["否", "是"]
        
        self.log(f"已设置{len(self.knowledge_questions)}个健康知识测试问题")
        return self.knowledge_questions
    
    def setup_information_database(self):
        """设置信息数据库，包含真实和虚假健康信息"""
        # 虚假信息
        rumors = [
            {"id": "r_1", "content": "最新发现：高血压患者只需每天喝一杯柠檬水，一个月就可以停药，不用再检查血压。", "type": "rumor"},
            {"id": "r_2", "content": "科学家证实：糖尿病患者服用肉桂粉一个月可以完全治愈糖尿病，不再需要胰岛素。", "type": "rumor"},
            {"id": "r_3", "content": "震惊医学界：研究表明老年人关节疼痛只是缺钙，每天吃一把钙片就能根除关节炎。", "type": "rumor"},
            {"id": "r_4", "content": "最新医学突破：老年痴呆症可以通过每天听音乐治愈，三个月内记忆力可恢复正常。", "type": "rumor"},
            {"id": "r_5", "content": "权威专家发现：高血脂患者只要每天喝醋，一周内胆固醇可以降到正常水平，不需要服药。", "type": "rumor"}
        ]
        
        # 真实信息
        official = [
            {"id": "o_1", "content": "国家卫健委提醒：糖尿病患者应遵医嘱用药，定期复查血糖，合理饮食和适当运动有助于辅助治疗。", "type": "official"},
            {"id": "o_2", "content": "心脏病专家提醒：高血压患者应坚持按医嘱服药，定期监测血压，保持健康生活方式。", "type": "official"},
            {"id": "o_3", "content": "权威医学期刊报道：适量运动可以改善老年人关节灵活性，但关节炎治疗需要综合方法。", "type": "official"},
            {"id": "o_4", "content": "神经科学研究表明：认知活动、社交互动和健康饮食可能有助于延缓认知衰退，但无法逆转已发生的老年痴呆。", "type": "official"},
            {"id": "o_5", "content": "心血管专家共识：降低胆固醇需要综合措施，包括饮食控制、运动和必要时的药物治疗，应在医生指导下进行。", "type": "official"}
        ]
        
        self.information_database = rumors + official
        self.log(f"已设置信息数据库：{len(rumors)}条虚假信息和{len(official)}条真实信息")
        return self.information_database
    
    def create_initial_users(self):
        """创建初始用户组"""
        for i in range(self.initial_users):
            user_id = f"{random.randint(0, 99999):05d}"
            health_knowledge = random.randint(0, 100)
            
            user = {
                "id": user_id,
                "health_knowledge": health_knowledge,
                "transmission_tendency": random.randint(0, 100) / 100.0,
                "rumor_sensitivity": 100 - health_knowledge,
                "social_circle_size": random.randint(10, 100),
                "upper_person": None,
                "generation": 0,  # 初始用户是第0代
                "created_at": self.current_time_step,
                "received_messages": [],
                "forwarded_messages": [],
                "verified_messages": [],
                "knowledge_tested": False
            }
            
            self.users[user_id] = user
            self.social_network.add_node(user_id, **user)
            
        self.log(f"已创建{self.initial_users}个初始用户")
        return list(self.users.keys())
    
    def test_user_knowledge(self, user_id):
        """测试用户的健康知识水平"""
        if not self.knowledge_questions:
            self.setup_knowledge_questions()
            
        user = self.users[user_id]
        if user["knowledge_tested"]:
            return user["health_knowledge"]
            
        # 模拟答题过程
        correct_count = 0
        expected_correct = user["health_knowledge"] / 10  # 根据当前知识水平预期正确数
        
        # 模拟随机答题，但向预期正确率靠拢
        for q in self.knowledge_questions:
            # 根据用户健康知识水平调整正确概率
            correct_prob = min(0.5 + user["health_knowledge"] / 200, 0.95)
            if random.random() < correct_prob:
                correct_count += 1
                
        # 更新用户健康知识水平
        new_knowledge_level = correct_count * 10
        old_knowledge_level = user["health_knowledge"]
        user["health_knowledge"] = new_knowledge_level
        user["rumor_sensitivity"] = 100 - new_knowledge_level
        user["knowledge_tested"] = True
        
        # 更新网络中的节点属性
        self.social_network.nodes[user_id]["health_knowledge"] = new_knowledge_level
        self.social_network.nodes[user_id]["rumor_sensitivity"] = 100 - new_knowledge_level
        
        self.log(f"用户{user_id}知识测试完成，从{old_knowledge_level}更新到{new_knowledge_level}")
        return new_knowledge_level

    def initialize_llm(self):
        """初始化LLM客户端"""
        if self.use_llm and self.llm_api_key:
            try:
                from llm import ZhipuLLM
                
                # 创建配置
                llm_config = {
                    'LLM_config': {
                        'API_KEY': self.llm_api_key,
                        'model_name': self.model_name
                    }
                }
                
                self.llm = ZhipuLLM(llm_config)
                self.log(f"LLM客户端初始化成功，使用模型: {self.model_name}")
                return True
            except Exception as e:
                self.log(f"LLM客户端初始化失败: {str(e)}", level="ERROR")
                self.use_llm = False
                return False
        return False

    def run_simulation(self):
        """运行完整的谣言传播模拟"""
        # 初始化
        self.log("开始初始化模拟...")
        self.setup_knowledge_questions()
        self.setup_information_database()
        self.create_initial_users()
        
        # 初始化LLM
        if self.use_llm:
            self.initialize_llm()
        
        # 测试用户知识水平
        self.log("开始测试用户知识水平...")
        for user_id in list(self.users.keys()):
            if self.use_llm:
                from llm import call_llm_for_knowledge_test
                call_llm_for_knowledge_test(self, user_id)
            else:
                self.test_user_knowledge(user_id)
        
        # 创建社交网络
        self.log("开始创建社交网络...")
        self.setup_social_network()
        
        # 开始模拟时间步
        self.log(f"开始模拟，最大时间步: {self.max_time_steps}")
        for t in range(self.max_time_steps):
            self.current_time_step = t
            self.log(f"===== 时间步 {t+1}/{self.max_time_steps} =====")
            
            # 传播信息
            self.spread_information(t)
            
            # 检查是否需要激活辟谣机器人
            if self.should_activate_debunking_bot(t):
                self.activate_debunking_bot()
            
            # 保存当前状态
            self.save_simulation_state(t)
            
            # 可视化当前状态
            if t % 5 == 0 or t == self.max_time_steps - 1:
                self.visualize_network(t)
        
        # 生成最终报告
        self.log("模拟完成，生成最终报告...")
        self.generate_final_report()
        
        return self.get_simulation_results()

    def simulate_message_decision(self, user_id, message):
        """当LLM不可用时模拟用户对消息的决策"""
        user = self.users[user_id]
        
        # 基于用户特征计算验证和转发概率
        # 健康知识越高，越可能验证消息
        verify_prob = user["health_knowledge"] / 100
        
        # 谣言敏感度越高，越不可能验证消息
        verify_prob -= user["rumor_sensitivity"] / 200  # 减少影响
        
        # 传播倾向越高，越可能转发消息
        forward_prob = user["transmission_tendency"]
        
        # 如果是谣言，谣言敏感度越高，越可能转发
        if message["type"] == "rumor":
            forward_prob += user["rumor_sensitivity"] / 200  # 增加影响
        
        # 确保概率在[0,1]范围内
        verify_prob = max(0, min(verify_prob, 1))
        forward_prob = max(0.3, min(forward_prob, 0.8))  # 设置最小转发概率为30%，最大为80%
        
        # 随机决定是否验证和转发
        verify = random.random() < verify_prob
        forward = random.random() < forward_prob
        
        return {
            "verify": verify,
            "forward": forward,
            "text": f"模拟决策: {'验证' if verify else '不验证'}, {'转发' if forward else '不转发'}"
        }

    def process_message_decision(self, user_id, message):
        """处理用户对消息的决策"""
        if self.use_llm:
            from llm import call_llm_for_message_decision
            decision = call_llm_for_message_decision(self, user_id, message)
        else:
            from llm import simulate_message_decision
            decision = simulate_message_decision(self, user_id, message)
        
        return decision

    def process_debunk_reaction(self, user_id, debunk_message, original_rumor):
        """处理用户对辟谣信息的反应"""
        if self.use_llm:
            from llm import call_llm_for_debunk_reaction
            reaction = call_llm_for_debunk_reaction(self, user_id, debunk_message, original_rumor)
        else:
            from llm import simulate_debunk_reaction
            reaction = simulate_debunk_reaction(self, user_id)
        
        return reaction

    def setup_social_network(self):
        """设置社交网络"""
        self.log("设置社交网络...")
        
        # 创建一个有向图
        self.social_network = nx.DiGraph()
        
        # 添加节点
        for user_id, user in self.users.items():
            self.social_network.add_node(user_id, 
                                         health_knowledge=user["health_knowledge"],
                                         rumor_sensitivity=user["rumor_sensitivity"],
                                         social_circle_size=user["social_circle_size"])
        
        # 添加边（社交连接）
        for user_id, user in self.users.items():
            # 根据用户的社交圈大小确定连接数量
            # 增加连接数量，使网络更密集
            num_connections = max(int(user["social_circle_size"] * 0.5), 5)  # 至少5个连接
            
            # 可能的连接对象（除了自己）
            possible_connections = [u for u in self.users.keys() if u != user_id]
            
            # 如果可能的连接对象不足，使用所有可能的连接
            actual_connections = min(num_connections, len(possible_connections))
            
            # 随机选择连接
            connections = random.sample(possible_connections, actual_connections)
            
            # 添加边
            for connection in connections:
                self.social_network.add_edge(user_id, connection)
        
        self.log(f"社交网络设置完成，共有 {self.social_network.number_of_nodes()} 个节点和 {self.social_network.number_of_edges()} 条边")
        
        # 检查网络连通性
        if not nx.is_strongly_connected(self.social_network):
            self.log("警告：社交网络不是强连通的，可能会影响信息传播", level="WARNING")
            
            # 确保网络至少是弱连通的
            if not nx.is_weakly_connected(self.social_network):
                self.log("警告：社交网络甚至不是弱连通的，正在添加额外连接", level="WARNING")
                
                # 找出所有连通分量
                components = list(nx.weakly_connected_components(self.social_network))
                
                # 如果有多个连通分量，添加它们之间的连接
                if len(components) > 1:
                    for i in range(len(components) - 1):
                        # 从每个连通分量中随机选择一个节点
                        node1 = random.choice(list(components[i]))
                        node2 = random.choice(list(components[i + 1]))
                        
                        # 添加双向连接
                        self.social_network.add_edge(node1, node2)
                        self.social_network.add_edge(node2, node1)
                        
                        self.log(f"添加连接：{node1} <-> {node2}")
        
        return self.social_network

    def spread_information(self, time_step):
        """在社交网络中传播信息"""
        self.log(f"时间步 {time_step+1}: 开始传播信息...")
        
        # 如果是第一个时间步，随机选择一些用户接收初始信息
        if time_step == 0:
            # 增加初始接收者比例
            initial_receivers_count = max(int(len(self.users) * 0.2), 10)  # 增加到20%
            initial_receivers = random.sample(list(self.users.keys()), min(initial_receivers_count, len(self.users)))
            
            # 随机选择一些谣言
            rumors = [info for info in self.information_database if info["type"] == "rumor"]
            if rumors:
                for user_id in initial_receivers:
                    # 为每个初始用户随机选择1-3条不同的谣言
                    selected_rumors = random.sample(rumors, min(random.randint(1, 3), len(rumors)))
                    
                    for rumor in selected_rumors:
                        # 创建消息
                        message = {
                            "id": f"msg_{len(self.messages_in_network)}",
                            "content": rumor["content"],
                            "type": rumor["type"],
                            "original_id": rumor["id"],
                            "sender": "system",
                            "receiver": user_id,
                            "time_step": time_step,
                            "verified": False,
                            "forwarded": False
                        }
                        
                        # 添加到网络中的消息列表
                        self.messages_in_network.append(message)
                        
                        # 添加到用户的接收消息列表
                        self.users[user_id]["received_messages"].append(message["id"])
                        
                        self.log(f"用户 {user_id} 收到初始谣言: {rumor['id']}")
        
        # 处理用户对已接收消息的反应
        messages_to_process = []
        for message in self.messages_in_network:
            # 只处理上一个时间步接收但尚未处理的消息
            if message["time_step"] == time_step and not message.get("processed", False):
                messages_to_process.append(message)
        
        for message in messages_to_process:
            receiver_id = message["receiver"]
            
            # 用户决定是否验证和转发
            decision = self.process_message_decision(receiver_id, message)
            
            # 更新消息状态
            message["verified"] = decision["verify"]
            message["forwarded"] = decision["forward"]
            message["processed"] = True
            
            # 如果用户决定转发
            if decision["forward"]:
                # 获取用户的社交连接
                connections = list(self.social_network.successors(receiver_id))
                
                # 如果没有连接，跳过
                if not connections:
                    continue
                
                # 决定转发给多少人（基于传播倾向）
                user = self.users[receiver_id]
                forward_count = max(1, int(len(connections) * user["transmission_tendency"]))
                forward_targets = random.sample(connections, min(forward_count, len(connections)))
                
                # 转发消息
                for target_id in forward_targets:
                    # 创建新消息
                    forwarded_message = {
                        "id": f"msg_{len(self.messages_in_network)}",
                        "content": message["content"],
                        "type": message["type"],
                        "original_id": message["original_id"],
                        "sender": receiver_id,
                        "receiver": target_id,
                        "time_step": time_step + 1,
                        "verified": False,
                        "forwarded": False,
                        "processed": False
                    }
                    
                    # 添加到网络中的消息列表
                    self.messages_in_network.append(forwarded_message)
                    
                    # 添加到用户的接收消息列表
                    self.users[target_id]["received_messages"].append(forwarded_message["id"])
                    
                    # 添加到发送者的转发消息列表
                    self.users[receiver_id]["forwarded_messages"].append(forwarded_message["id"])
                    
                    self.log(f"用户 {receiver_id} 将消息 {message['id']} 转发给用户 {target_id}")
        
        # 统计当前谣言传播情况
        rumor_count = sum(1 for msg in self.messages_in_network if msg["type"] == "rumor" and msg["forwarded"])
        total_messages = len(self.messages_in_network)
        
        self.log(f"时间步 {time_step+1} 传播完成: 总消息数 {total_messages}, 谣言转发数 {rumor_count}")

    def should_activate_debunking_bot(self, time_step):
        """判断是否应该激活辟谣机器人"""
        # 如果辟谣机器人已经激活，直接返回False
        if self.debunking_bot_active:
            return False
        
        # 计算收到谣言的用户比例
        users_received_rumor = set()
        for msg in self.messages_in_network:
            if msg["type"] == "rumor" and not msg.get("deleted", False):
                users_received_rumor.add(msg["receiver"])
        
        rumor_ratio = len(users_received_rumor) / len(self.users) if self.users else 0
        
        # 降低阈值到15%
        threshold = 0.15
        should_activate = rumor_ratio >= threshold
        
        if should_activate:
            self.log(f"时间步 {time_step+1}: 谣言传播比例达到 {rumor_ratio:.2f}，超过阈值 {threshold}，将激活辟谣机器人")
        
        return should_activate

    def activate_debunking_bot(self):
        """激活辟谣机器人，发布辟谣信息"""
        self.log("激活辟谣机器人...")
        
        # 标记辟谣机器人为激活状态
        self.debunking_bot_active = True
        
        # 获取所有已传播的谣言
        spread_rumors = {}
        for msg in self.messages_in_network:
            if msg["type"] == "rumor" and msg["forwarded"]:
                original_id = msg["original_id"]
                if original_id not in spread_rumors:
                    spread_rumors[original_id] = 0
                spread_rumors[original_id] += 1
        
        # 如果没有传播的谣言，返回
        if not spread_rumors:
            self.log("没有需要辟谣的谣言")
            return
        
        # 按传播程度排序谣言
        sorted_rumors = sorted(spread_rumors.items(), key=lambda x: x[1], reverse=True)
        
        # 获取传播最广的谣言
        top_rumor_id = sorted_rumors[0][0]
        
        # 找到对应的谣言和辟谣信息
        rumor = None
        debunk = None
        
        for info in self.information_database:
            if info["id"] == top_rumor_id:
                rumor = info
            elif info["type"] == "official" and info["id"] == "o_" + top_rumor_id[2:]:
                debunk = info
        
        # 如果没有找到对应的辟谣信息，创建一个
        if not debunk and rumor:
            debunk_id = "o_" + top_rumor_id[2:]
            debunk_content = f"官方辟谣：近期流传的'{rumor['content']}'是错误信息。请遵循医生建议，不要轻信网络传言。"
            
            debunk = {
                "id": debunk_id,
                "content": debunk_content,
                "type": "official"
            }
            
            # 添加到信息数据库
            self.information_database.append(debunk)
        
        # 如果找到了谣言和辟谣信息，向所有用户发送辟谣信息
        if rumor and debunk:
            self.log(f"发布辟谣信息: {debunk['id']}")
            
            # 向所有用户发送辟谣信息
            for user_id in self.users.keys():
                # 创建辟谣消息
                debunk_message = {
                    "id": f"msg_{len(self.messages_in_network)}",
                    "content": debunk["content"],
                    "type": "official",
                    "original_id": debunk["id"],
                    "sender": "debunk_bot",
                    "receiver": user_id,
                    "time_step": self.current_time_step,
                    "verified": True,
                    "forwarded": False,
                    "processed": False,
                    "is_debunk": True,
                    "debunk_for": rumor["id"]
                }
                
                # 添加到网络中的消息列表
                self.messages_in_network.append(debunk_message)
                
                # 添加到用户的接收消息列表
                self.users[user_id]["received_messages"].append(debunk_message["id"])
                
                # 处理用户对辟谣信息的反应
                reaction = self.process_debunk_reaction(user_id, debunk_message, rumor)
                
                # 如果用户改变了想法并且之前转发过谣言
                if reaction["changed_mind"]:
                    # 查找用户转发的相关谣言
                    for msg_id in self.users[user_id]["forwarded_messages"]:
                        msg = next((m for m in self.messages_in_network if m["id"] == msg_id), None)
                        if msg and msg["original_id"] == rumor["id"]:
                            # 根据用户的反应采取行动
                            if reaction["action"] == "删除":
                                # 标记消息为已删除
                                msg["deleted"] = True
                                self.log(f"用户 {user_id} 删除了转发的谣言 {msg_id}")
                            elif reaction["action"] == "发布更正":
                                # 创建更正消息
                                correction_message = {
                                    "id": f"msg_{len(self.messages_in_network)}",
                                    "content": f"更正：我之前转发的信息'{rumor['content']}'是错误的。{debunk['content']}",
                                    "type": "correction",
                                    "original_id": debunk["id"],
                                    "sender": user_id,
                                    "time_step": self.current_time_step,
                                    "verified": True,
                                    "forwarded": True,
                                    "processed": True,
                                    "is_correction": True,
                                    "correction_for": msg_id
                                }
                                
                                # 向之前接收谣言的用户发送更正
                                for fwd_msg_id in self.users[user_id]["forwarded_messages"]:
                                    fwd_msg = next((m for m in self.messages_in_network if m["id"] == fwd_msg_id), None)
                                    if fwd_msg and fwd_msg["original_id"] == rumor["id"]:
                                        # 创建发送给特定接收者的更正消息
                                        receiver_correction = correction_message.copy()
                                        receiver_correction["id"] = f"msg_{len(self.messages_in_network)}"
                                        receiver_correction["receiver"] = fwd_msg["receiver"]
                                        
                                        # 添加到网络中的消息列表
                                        self.messages_in_network.append(receiver_correction)
                                        
                                        # 添加到接收者的接收消息列表
                                        self.users[fwd_msg["receiver"]]["received_messages"].append(receiver_correction["id"])
                                        
                                        self.log(f"用户 {user_id} 向用户 {fwd_msg['receiver']} 发送了更正信息")

    def save_simulation_state(self, time_step):
        """保存当前模拟状态"""
        # 创建状态快照
        state = {
            "time_step": time_step,
            "users": self.users,
            "messages": self.messages_in_network,
            "debunking_bot_active": self.debunking_bot_active
        }
        
        # 保存到文件
        state_file = os.path.join(self.save_dir, f"state_step_{time_step}.json")
        with open(state_file, 'w', encoding='utf-8') as f:
            # 将复杂对象转换为可序列化的形式
            import json
            
            # 创建可序列化的副本
            serializable_state = {
                "time_step": state["time_step"],
                "users": {k: {kk: vv for kk, vv in v.items() if isinstance(vv, (str, int, float, bool, list, dict))} 
                         for k, v in state["users"].items()},
                "messages": [{k: v for k, v in msg.items() if isinstance(v, (str, int, float, bool, list, dict))} 
                            for msg in state["messages"]],
                "debunking_bot_active": state["debunking_bot_active"]
            }
            
            json.dump(serializable_state, f, ensure_ascii=False, indent=2)
        
        self.log(f"已保存时间步 {time_step} 的模拟状态")

    def visualize_network(self, time_step):
        """可视化当前社交网络状态"""
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            
            # 创建图形
            plt.figure(figsize=(12, 8))
            
            # 获取节点颜色（基于健康知识水平）
            node_colors = []
            for node in self.social_network.nodes():
                knowledge = self.social_network.nodes[node].get("health_knowledge", 50)
                # 颜色从红色（低知识）到绿色（高知识）
                color = (1 - knowledge/100, knowledge/100, 0)
                node_colors.append(color)
            
            # 获取节点大小（基于社交圈大小）
            node_sizes = []
            for node in self.social_network.nodes():
                size = self.social_network.nodes[node].get("social_circle_size", 10)
                node_sizes.append(size * 10)
            
            # 获取边颜色（基于消息类型）
            edge_colors = []
            for u, v in self.social_network.edges():
                # 查找从u到v的消息
                messages = [msg for msg in self.messages_in_network 
                           if msg["sender"] == u and msg["receiver"] == v]
                
                if any(msg["type"] == "rumor" and not msg.get("deleted", False) for msg in messages):
                    # 谣言消息为红色
                    edge_colors.append('red')
                elif any(msg["type"] == "official" or msg["type"] == "correction" for msg in messages):
                    # 辟谣或更正消息为绿色
                    edge_colors.append('green')
                else:
                    # 默认为灰色
                    edge_colors.append('grey')
            
            # 绘制网络
            pos = nx.spring_layout(self.social_network, seed=42)
            nx.draw_networkx_nodes(self.social_network, pos, node_color=node_colors, node_size=node_sizes)
            nx.draw_networkx_edges(self.social_network, pos, edge_color=edge_colors, width=1.5, alpha=0.7)
            nx.draw_networkx_labels(self.social_network, pos, font_size=8)
            
            # 添加标题和图例
            plt.title(f"Rumor propagation network - time step {time_step}")
            
            # 保存图像
            plt.savefig(os.path.join(self.save_dir, f"network_step_{time_step}.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            self.log(f"已保存时间步 {time_step} 的网络可视化")
        except Exception as e:
            self.log(f"网络可视化失败: {str(e)}", level="ERROR")

    def generate_final_report(self):
        """生成最终模拟报告"""
        # 计算统计数据
        total_messages = len(self.messages_in_network)
        rumor_messages = [msg for msg in self.messages_in_network if msg["type"] == "rumor"]
        official_messages = [msg for msg in self.messages_in_network if msg["type"] == "official"]
        correction_messages = [msg for msg in self.messages_in_network if msg.get("type") == "correction"]
        
        # 计算谣言传播率
        users_received_rumor = set()
        for msg in rumor_messages:
            if not msg.get("deleted", False):
                users_received_rumor.add(msg["receiver"])
        
        rumor_spread_ratio = len(users_received_rumor) / len(self.users) if self.users else 0
        
        # 计算辟谣效果
        users_changed_mind = 0
        for user_id, user in self.users.items():
            # 检查用户是否收到过辟谣信息并改变了想法
            for msg_id in user["received_messages"]:
                msg = next((m for m in self.messages_in_network if m["id"] == msg_id), None)
                if msg and msg.get("is_debunk", False):
                    # 检查用户是否删除或更正了谣言
                    for fwd_msg_id in user.get("forwarded_messages", []):
                        fwd_msg = next((m for m in self.messages_in_network if m["id"] == fwd_msg_id), None)
                        if fwd_msg and fwd_msg["original_id"] == msg.get("debunk_for") and (
                            fwd_msg.get("deleted", False) or 
                            any(cm.get("correction_for") == fwd_msg["id"] for cm in correction_messages)
                        ):
                            users_changed_mind += 1
                            break
        
        debunk_effectiveness = users_changed_mind / len(users_received_rumor) if users_received_rumor else 0
        
        # 创建报告
        report = {
            "总用户数": len(self.users),
            "总消息数": total_messages,
            "谣言消息数": len(rumor_messages),
            "官方消息数": len(official_messages),
            "更正消息数": len(correction_messages),
            "谣言传播率": rumor_spread_ratio,
            "辟谣效果": debunk_effectiveness,
            "辟谣机器人是否激活": self.debunking_bot_active,
            "模拟时间步数": self.current_time_step + 1
        }
        
        # 保存报告
        report_file = os.path.join(self.save_dir, "final_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            import json
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 同时保存为文本格式
        report_txt = os.path.join(self.save_dir, "final_report.txt")
        with open(report_txt, 'w', encoding='utf-8') as f:
            f.write("谣言传播模拟最终报告\n")
            f.write("=" * 30 + "\n\n")
            
            for key, value in report.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.2f}\n")
                else:
                    f.write(f"{key}: {value}\n")
        
        self.log("已生成最终报告")
        
        return report

    def get_simulation_results(self):
        """获取模拟结果"""
        # 返回最终报告数据
        return self.generate_final_report()