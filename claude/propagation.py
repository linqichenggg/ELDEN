import random
import time

def expand_network(self):
    """根据现有用户的社交圈大小扩展网络"""
    new_users = []
    
    for user_id, user in list(self.users.items()):
        if self.social_network.out_degree(user_id) >= user["social_circle_size"]:
            continue  # 已经达到社交圈大小上限
            
        # 计算还需要添加的朋友数量
        friends_to_add = user["social_circle_size"] - self.social_network.out_degree(user_id)
        
        for _ in range(friends_to_add):
            # 创建新用户
            new_id = f"{random.randint(0, 99999):05d}"
            while new_id in self.users:
                new_id = f"{random.randint(0, 99999):05d}"
                
            new_user = {
                "id": new_id,
                "health_knowledge": random.randint(0, 100),
                "transmission_tendency": random.randint(0, 100) / 100.0,
                "rumor_sensitivity": None,  # 会在测试后更新
                "social_circle_size": random.randint(10, 100),
                "upper_person": user_id,
                "generation": user["generation"] + 1,  # 增加一代
                "created_at": self.current_time_step,
                "received_messages": [],
                "forwarded_messages": [],
                "verified_messages": [],
                "knowledge_tested": False
            }
            
            # 先添加到社交网络
            self.social_network.add_node(new_id, **new_user)
            self.social_network.add_edge(user_id, new_id)
            
            # 然后添加到用户字典
            self.users[new_id] = new_user
            
            # 使用LLM测试用户健康知识
            if self.use_llm and self.llm:
                new_knowledge_level = self.call_llm_for_knowledge_test(new_id)
            else:
                # 使用随机模拟
                new_knowledge_level = self.test_user_knowledge(new_id)
                
            new_user["rumor_sensitivity"] = 100 - new_knowledge_level
            
            # 确保网络节点属性更新
            self.social_network.nodes[new_id]["health_knowledge"] = new_knowledge_level
            self.social_network.nodes[new_id]["rumor_sensitivity"] = 100 - new_knowledge_level
            
            new_users.append(new_id)
            
    self.log(f"网络扩展完成，添加了{len(new_users)}个新用户")
    return new_users

def select_initial_messages(self):
    """为初始用户选择要传播的信息"""
    if not self.information_database:
        self.setup_information_database()
        
    selected_messages = []
    
    # 随机选择虚假和真实信息的混合
    rumors = [info for info in self.information_database if info["type"] == "rumor"]
    official = [info for info in self.information_database if info["type"] == "official"]
    
    # 按照1/9的虚假/官方比例选择
    num_rumors = max(1, len(rumors) // 9)
    num_official = len(official) - num_rumors
    
    selected_rumors = random.sample(rumors, min(num_rumors, len(rumors)))
    selected_official = random.sample(official, min(num_official, len(official)))
    
    selected_messages = selected_rumors + selected_official
    random.shuffle(selected_messages)
    
    self.log(f"为初始传播选择了{len(selected_messages)}条信息 ({len(selected_rumors)}虚假/{len(selected_official)}真实)")
    return selected_messages

def simulate_message_propagation(self, time_steps=None):
    """模拟信息在网络中的传播"""
    if time_steps is None:
        time_steps = self.max_time_steps
    
    # 检查LLM状态
    if hasattr(self, 'use_llm') and self.use_llm and not self.llm:
        self.log("警告：设置了使用LLM但LLM客户端未初始化，将使用随机模拟", level="WARNING")
        self.use_llm = False
    
    # 记录LLM使用状态
    llm_status = "使用" if hasattr(self, 'use_llm') and self.use_llm else "不使用"
    self.log(f"开始模拟信息传播 ({llm_status} LLM)")
        
    # 初始化消息传播
    if self.current_time_step == 0:
        # 创建初始用户和信息
        if not self.users:
            self.create_initial_users()
        if not self.information_database:
            self.setup_information_database()
            
        # 为初始用户分配要传播的消息
        initial_messages = self.select_initial_messages()
        for user_id in list(self.users.keys())[:self.initial_users]:
            # 如果启用LLM，先测试用户知识水平
            if hasattr(self, 'use_llm') and self.use_llm and self.llm:
                try:
                    self.call_llm_for_knowledge_test(user_id)
                    self.log(f"用户 {user_id} 使用LLM完成知识测试")
                except Exception as e:
                    self.log(f"LLM知识测试失败: {str(e)}", level="ERROR")
                    self.test_user_knowledge(user_id)
            else:
                self.test_user_knowledge(user_id)
                
            for msg in initial_messages:
                message_instance = {
                    "id": f"{msg['id']}_{int(time.time())}_{random.randint(1000, 9999)}",
                    "content": msg["content"],
                    "type": msg["type"],
                    "original_id": msg["id"],
                    "sender": user_id,
                    "created_at": self.current_time_step,
                    "path": [user_id],
                    "verified_by": []
                }
                
                # 模拟用户决策
                self.users[user_id]["received_messages"].append(message_instance["id"])
                
                # 使用LLM决策是否转发初始消息
                if self.use_llm and self.llm:
                    decision = self.call_llm_for_message_decision(user_id, message_instance)
                    if decision["forward"]:
                        self.users[user_id]["forwarded_messages"].append(message_instance["id"])
                        self.messages_in_network.append(message_instance)
                    if decision["verify"]:
                        self.users[user_id]["verified_messages"].append(message_instance["id"])
                        message_instance["verified_by"].append(user_id)
                else:
                    # 假设初始用户直接转发
                    self.users[user_id]["forwarded_messages"].append(message_instance["id"])
                    self.messages_in_network.append(message_instance)
    
    # 主模拟循环
    for step in range(time_steps):
        self.current_time_step += 1
        self.log(f"=== 时间步骤 {self.current_time_step} ===")
        
        # 扩展网络
        if self.current_time_step <= 3:  # 只在前几个时间步骤扩展网络
            self.expand_network()
            
        # 收集当前可传播的消息
        messages_to_propagate = []
        for msg in self.messages_in_network:
            sender = msg["sender"]
            if sender in self.users and msg["id"] in self.users[sender]["forwarded_messages"]:
                messages_to_propagate.append(msg)
        
        # 传播消息
        new_messages = []
        for msg in messages_to_propagate:
            sender = msg["sender"]
            # 获取发送者的所有朋友
            neighbors = list(self.social_network.successors(sender))
            
            for receiver in neighbors:
                # 避免重复接收
                if any(m["id"] == msg["id"] for m in 
                       [m for m in self.messages_in_network if m["id"] in self.users[receiver]["received_messages"]]):
                    continue
                
                # 创建消息实例
                message_instance = msg.copy()
                message_instance["id"] = f"{msg['original_id']}_{int(time.time())}_{random.randint(1000, 9999)}"
                message_instance["sender"] = sender
                message_instance["path"] = msg["path"] + [receiver]
                
                # 用户接收消息
                self.users[receiver]["received_messages"].append(message_instance["id"])
                
                # 用户决策是否验证和转发
                try:
                    if hasattr(self, 'use_llm') and self.use_llm and self.llm:
                        decision = self.call_llm_for_message_decision(receiver, message_instance)
                        self.log(f"用户 {receiver} 使用LLM做出决策: 验证={decision['verify']}, 转发={decision['forward']}")
                    else:
                        decision = self.simulate_message_decision(receiver, message_instance)
                except Exception as e:
                    self.log(f"决策过程出错: {str(e)}", level="ERROR")
                    decision = self.simulate_message_decision(receiver, message_instance)
                
                if decision["verify"]:
                    self.users[receiver]["verified_messages"].append(message_instance["id"])
                    message_instance["verified_by"].append(receiver)
                    
                    # 如果是谣言且用户验证了，大概率不会转发
                    if message_instance["type"] == "rumor":
                        if random.random() > 0.2:  # 80%概率不转发已验证的谣言
                            decision["forward"] = False
                
                if decision["forward"]:
                    self.users[receiver]["forwarded_messages"].append(message_instance["id"])
                    new_messages.append(message_instance)
        
        # 添加新消息到网络
        self.messages_in_network.extend(new_messages)
        
        # 检查是否需要激活辟谣机器人
        if self.current_time_step >= 3 and not self.debunking_bot_active:
            rumor_ratio = self.calculate_rumor_ratio()
            if rumor_ratio >= self.debunking_threshold:
                self.activate_debunking_bot()
        
        # 输出当前状态
        total_messages = len(self.messages_in_network)
        forwarded_messages = sum(len(user["forwarded_messages"]) for user in self.users.values())
        verified_messages = sum(len(user["verified_messages"]) for user in self.users.values())
        
        self.log(f"当前消息总数: {total_messages}, 转发: {forwarded_messages}, 验证: {verified_messages}")
        self.log(f"谣言比例: {self.calculate_rumor_ratio():.2f}")
        
        # 检查终止条件
        if self.check_termination_condition():
            self.log("模拟达到终止条件，提前结束")
            break
            
    # 保存模拟结果
    self.save_simulation_results()