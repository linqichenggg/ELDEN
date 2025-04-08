import mesa
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import random
import pickle
import networkx as nx
from citizen import Citizen
from tqdm import tqdm
from datetime import datetime
from utils import (
    update_day,
    clear_cache,
    create_social_network,
    load_real_users,
    DialogueState,
    should_stop_dialogue,
    calculate_final_belief_change,
    format_dialogue_history,
    get_dialogue_summary,
    create_memory_from_health_opinion
)
from prompt import *
import json
import psutil
import os

# 简化后的数据收集函数
def compute_num_susceptible(model):
    return sum([1 for a in model.schedule.agents if a.health_condition == "Susceptible"])

def compute_num_infected(model):
    return sum([1 for a in model.schedule.agents if a.health_condition == "Infected"])

def compute_num_recovered(model):
    return sum([1 for a in model.schedule.agents if a.health_condition == "Recovered"])

class World(mesa.Model):
    '''
    The world where Citizens exist
    '''
    def __init__(self, args, initial_healthy=18, initial_infected=2, contact_rate=3):
        # 初始化基本参数
        self.initial_healthy = initial_healthy
        self.initial_infected = initial_infected
        self.population = initial_healthy + initial_infected
        self.step_count = args.no_days
        self.offset = 0  # 检查点加载的偏移量
        self.name = args.name
        self.topic = "每天喝一杯白酒可以预防糖尿病和高血压，对老年人的心脑血管健康有益"
        
        # 感染相关变量
        self.infected = initial_infected
        self.susceptible = initial_healthy
        self.recovered = 0
        
        # 每日计数器
        self.daily_new_infected_cases = 0
        self.daily_new_susceptible_cases = 0
        self.daily_new_recovered_cases = 0
        
        # 兼容旧代码的变量
        self.total_contact_rates = 0
        self.daily_contact_count = 0
        self.track_contact_rate = [0]
        self.list_new_infected_cases = [self.initial_infected]
        self.list_new_susceptible_cases = [0]
        self.list_new_recovered_cases = [0]
        
        self.current_date = datetime(2025, 6, 1)
        self.contact_rate = args.contact_rate   
        
        # 初始化调度器
        self.schedule = RandomActivation(self)  

        # 初始化数据收集器
        self.datacollector = DataCollector(
            model_reporters={
                "Susceptible": compute_num_susceptible,
                "Infected": compute_num_infected,
                "Recovered": compute_num_recovered,
            })
        
        # 加载真实用户数据
        try:
            real_users = load_real_users(args.user_data_file)
            if len(real_users) < self.population:
                raise ValueError(f"真实用户数据不足。需要 {self.population} 个用户，但只有 {len(real_users)} 个。")
        except Exception as e:
            print(f"加载用户数据失败: {e}")
            raise

        # 初始化代理人
        for i in range(self.population):
            agent_id = i  # 简化ID分配
            
            # 获取真实用户数据
            user_data = real_users[i]
            
            # 创建健康或感染的代理人
            if i < self.initial_healthy:
                health_condition = "Susceptible"
                opinion = random.choice(topic_sentence_susceptible)
                # 添加更明确的不信标记
                opinion = "我相信：" + opinion
            else:
                health_condition = "Infected"
                opinion = random.choice(topic_sentence_infeted)
                # 添加更明确的相信标记
                opinion = "我相信：" + opinion
            
            # 从用户数据中提取必要信息
            user_name = user_data["name"]
            user_traits = user_data["traits"]
            user_education = user_data["education"]
            user_description = user_data["description"]
            user_health_opinion = user_data["health_opinion"]

            # 创建Citizen实例
            citizen = Citizen(
                model=self,
                unique_id=agent_id, 
                name=user_name, 
                age=random.randrange(60, 90),
                traits=user_traits, 
                opinion=opinion,
                qualification=user_education,
                health_condition=health_condition,
                topic=self.topic
            )
            
            # 设置额外属性
            citizen.self_description = user_description
            
            # 从健康观点创建长期记忆
            if user_health_opinion:
                # 确保long_opinion_memory是列表
                if not hasattr(citizen, 'long_opinion_memory') or citizen.long_opinion_memory is None:
                    citizen.long_opinion_memory = []
                elif isinstance(citizen.long_opinion_memory, str):
                    # 如果是字符串，转换为列表
                    citizen.long_opinion_memory = [citizen.long_opinion_memory]
                
                long_memory = create_memory_from_health_opinion(user_health_opinion, user_name)
                citizen.long_opinion_memory.append(long_memory)
            
            # 添加代理人到调度器
            self.schedule.add(citizen)

        # 在初始化代理人后，手动记录初始状态
        self.datacollector.collect(self)  # 收集初始状态
        
        # 新增：创建社交网络
        self.social_network = create_social_network(self.schedule.agents)
        
        # 新增：对话配对和记录
        self.dialogue_pairs = []  # 当前步骤的对话配对
        self.dialogue_records = []  # 所有对话记录
        
        # 新增：多轮对话参数
        self.max_dialogue_turns = 3  # 最大对话轮次
        self.dialogue_convergence_threshold = 0.1  # 对话收敛阈值
        
        # 初始化完成后检查一致性
        self.check_consistency()
        
        print(f"初始化完成: 人口={self.population}, 社交连接={self.social_network.number_of_edges()}")

    # 新增：决定代理人对话配对
    def decide_dialogue_pairs(self):
        '''决定代理人之间的对话配对'''
        # 清空当前对话配对
        self.dialogue_pairs = []
        
        # 获取所有代理人
        available_agents = list(self.schedule.agents)
        random.shuffle(available_agents)
        
        while len(available_agents) >= 2:
            agent1 = available_agents.pop(0)
            
            # 基于社交网络选择最可能的对话伙伴
            potential_partners = []
            for agent2 in available_agents:
                if self.social_network.has_edge(agent1.unique_id, agent2.unique_id):
                    # 获取连接强度
                    strength = self.social_network[agent1.unique_id][agent2.unique_id]['weight']
                    potential_partners.append((agent2, strength))
            
            # 按连接强度排序
            potential_partners.sort(key=lambda x: x[1], reverse=True)
            
            if potential_partners:
                # 优先选择社交网络中连接强度高的伙伴
                # 使用轮盘赌选择，连接强度越高概率越大
                total_strength = sum(strength for _, strength in potential_partners)
                r = random.uniform(0, total_strength)
                cumulative = 0
                selected_agent = None
                
                for agent, strength in potential_partners:
                    cumulative += strength
                    if r <= cumulative:
                        selected_agent = agent
                        break
                
                if not selected_agent:  # 以防万一
                    selected_agent = potential_partners[0][0]
            else:
                # 如果没有社交连接，随机选择
                selected_agent = random.choice(available_agents)
                
            available_agents.remove(selected_agent)
            
            # 创建对话配对
            self.dialogue_pairs.append((agent1, selected_agent))
            
            # 记录社交互动
            agent1.interaction_history.append(selected_agent.unique_id)
            selected_agent.interaction_history.append(agent1.unique_id)
            
            # 更新对话伙伴计数
            if selected_agent.unique_id in agent1.dialogue_partners:
                agent1.dialogue_partners[selected_agent.unique_id] += 1
            else:
                agent1.dialogue_partners[selected_agent.unique_id] = 1
                
            if agent1.unique_id in selected_agent.dialogue_partners:
                selected_agent.dialogue_partners[agent1.unique_id] += 1
            else:
                selected_agent.dialogue_partners[agent1.unique_id] = 1

    # 新增：执行多轮对话
    def conduct_dialogue(self, agent1, agent2):
        '''执行两个代理人之间的多轮对话'''
        print(f"开始对话: {agent1.name}(ID:{agent1.unique_id}) 与 {agent2.name}(ID:{agent2.unique_id})")
        
        # 初始化对话状态
        dialogue_state = DialogueState(
            topic=self.topic,
            agent1_id=agent1.unique_id,
            agent2_id=agent2.unique_id
        )
        
        # 初始化对话历史
        conversation_history = []
        
        # Agent1开始对话 - 添加错误处理
        try:
            response1_data = agent1.generate_dialogue_initiation(agent2)
            # 确保response1_data包含response键
            if "response" not in response1_data:
                print(f"警告：代理人{agent1.name}(ID:{agent1.unique_id})的初始响应缺少'response'键")
                response1_data["response"] = f"我是{agent1.name}，我想讨论一下关于{self.topic}的看法。"
        except Exception as e:
            print(f"错误：代理人{agent1.name}生成对话初始化失败: {e}")
            response1_data = {
                "response": f"我是{agent1.name}，我想讨论一下关于{self.topic}的看法。",
                "internal_thoughts": "生成初始响应时出错",
                "belief_shift": 0,
                "reasoning": "处理错误"
            }
        
        conversation_history.append({
            "speaker": agent1.name,
            "content": response1_data["response"],
            "turn": 0
        })
        
        # 更新对话状态
        dialogue_state.update_after_turn(agent1.unique_id, response1_data)
        
        # 执行多轮对话
        for turn in range(1, self.max_dialogue_turns + 1):
            # Agent2回应 - 添加错误处理
            try:
                response2_data = agent2.generate_dialogue_response(
                    conversation_history=conversation_history,
                    dialogue_state=dialogue_state,
                    other_agent=agent1
                )
                if "response" not in response2_data:
                    print(f"警告：代理人{agent2.name}(ID:{agent2.unique_id})的回应缺少'response'键")
                    response2_data["response"] = f"我是{agent2.name}，谢谢分享您的观点。我正在思考这个话题。"
            except Exception as e:
                print(f"错误：代理人{agent2.name}生成对话回应失败: {e}")
                response2_data = {
                    "response": f"我是{agent2.name}，谢谢分享您的观点。我正在思考这个话题。",
                    "internal_thoughts": "生成回应时出错",
                    "belief_shift": 0,
                    "reasoning": "处理错误"
                }
            
            conversation_history.append({
                "speaker": agent2.name,
                "content": response2_data["response"],
                "turn": turn * 2 - 1
            })
            
            # 更新对话状态
            dialogue_state.update_after_turn(agent2.unique_id, response2_data)
            
            # Agent1回应 - 添加错误处理
            try:
                response1_data = agent1.generate_dialogue_response(
                    conversation_history=conversation_history,
                    dialogue_state=dialogue_state,
                    other_agent=agent2
                )
                if "response" not in response1_data:
                    print(f"警告：代理人{agent1.name}(ID:{agent1.unique_id})的回应缺少'response'键")
                    response1_data["response"] = f"我是{agent1.name}，感谢您的回复。让我再思考一下这个问题。"
            except Exception as e:
                print(f"错误：代理人{agent1.name}生成对话回应失败: {e}")
                response1_data = {
                    "response": f"我是{agent1.name}，感谢您的回复。让我再思考一下这个问题。",
                    "internal_thoughts": "生成回应时出错",
                    "belief_shift": 0,
                    "reasoning": "处理错误"
                }
            
            conversation_history.append({
                "speaker": agent1.name,
                "content": response1_data["response"],
                "turn": turn * 2
            })
            
            # 更新对话状态
            dialogue_state.update_after_turn(agent1.unique_id, response1_data)
            
            # 检查停止条件
            try:
                should_stop = should_stop_dialogue(
                    dialogue_state, 
                    response1_data, 
                    response2_data,
                    max_turns=self.max_dialogue_turns,
                    convergence_threshold=self.dialogue_convergence_threshold
                )
                if should_stop:
                    print(f"对话在第{turn}轮结束，原因: {dialogue_state.stop_reason}")
                    break
            except Exception as e:
                print(f"错误：检查对话停止条件时失败: {e}")
                print(f"强制在第{turn}轮结束对话")
                dialogue_state.stop_reason = "错误处理导致对话结束"
                break
        
        # 计算最终信念变化 - 添加错误处理
        try:
            belief_change1 = calculate_final_belief_change(agent1, dialogue_state, conversation_history)
        except Exception as e:
            print(f"错误：计算代理人{agent1.name}的信念变化失败: {e}")
            belief_change1 = 0
        
        try:
            belief_change2 = calculate_final_belief_change(agent2, dialogue_state, conversation_history)
        except Exception as e:
            print(f"错误：计算代理人{agent2.name}的信念变化失败: {e}")
            belief_change2 = 0
        
        # 更新代理人信念 - 添加错误处理
        try:
            agent1.update_belief_after_dialogue(belief_change1, conversation_history)
        except Exception as e:
            print(f"错误：更新代理人{agent1.name}的信念失败: {e}")
        
        try:
            agent2.update_belief_after_dialogue(belief_change2, conversation_history)
        except Exception as e:
            print(f"错误：更新代理人{agent2.name}的信念失败: {e}")
        
        # 记录对话结果 - 确保包含详细对话内容
        dialogue_result = {
            "agents": (agent1.unique_id, agent2.unique_id),
            "agent_names": (agent1.name, agent2.name),  # 添加代理人名称以便于阅读
            "history": conversation_history,  # 确保这里包含完整对话历史
            "belief_changes": (belief_change1, belief_change2),
            "final_beliefs": (agent1.beliefs[-1], agent2.beliefs[-1]),
            "stop_reason": dialogue_state.stop_reason,
            "turns": dialogue_state.turn_count,
            "topic": self.topic  # 添加主题以便更好地理解对话内容
        }
        
        return conversation_history, belief_change1, belief_change2, dialogue_result

    def decide_agent_interactions(self):
        '''决定代理人之间的互动（兼容旧代码）'''
        # 基本互动
        for agent in self.schedule.agents:
            potential_interactions = [a for a in self.schedule.agents if a is not agent]  
            random.shuffle(potential_interactions) 
            potential_interactions = potential_interactions[:self.contact_rate]  
            for other_agent in potential_interactions:
                agent.agent_interaction.append(other_agent)    

    def step(self):
        '''模型时间步进'''
        # 决定代理人对话配对
        self.decide_dialogue_pairs()
        
        # 执行所有对话
        dialogue_results = []
        for agent1, agent2 in self.dialogue_pairs:
            # 执行多轮对话
            conversation_history, belief_change1, belief_change2, dialogue_result = self.conduct_dialogue(agent1, agent2)
            
            # 调试信息
            print(f"对话结束: {agent1.name} 与 {agent2.name}, {len(conversation_history)} 轮对话")
            if conversation_history:
                print(f"第一轮: {conversation_history[0]['speaker']}: {conversation_history[0]['content'][:50]}...")
                if len(conversation_history) > 1:
                    print(f"最后一轮: {conversation_history[-1]['speaker']}: {conversation_history[-1]['content'][:50]}...")
            
            dialogue_results.append(dialogue_result)
        
        # 记录对话结果
        self.dialogue_records.extend(dialogue_results)
        print(f"当前步骤结束，累计 {len(self.dialogue_records)} 条对话记录")
        
        # 更新所有代理人状态
        print("Updating agent days...")
        for agent in self.schedule.agents:
            update_day(agent)
            print(f"Agent {agent.unique_id}: {agent.health_condition} (belief={agent.beliefs[-1] if agent.beliefs else None})")

        # 在更新完所有代理人状态后检查一致性
        self.check_consistency()

        # 在world.py的step方法结束时
        belief_count = sum(a.beliefs[-1] for a in self.schedule.agents)
        infected_count = sum(1 for a in self.schedule.agents if a.health_condition == "Infected")
        if belief_count != infected_count:
            print(f"WARNING: Belief count ({belief_count}) does not match infected count ({infected_count})")

        # 兼容旧代码的接触率记录
        self.track_contact_rate.append(len(self.dialogue_pairs) * 2)
        print(f"步骤结束: 对话配对数={len(self.dialogue_pairs)}, 总人口={self.population}")

    def run_model(self, checkpoint_path=None, offset=0):
        """
        运行模型
        
        参数:
        checkpoint_path: 检查点保存路径
        offset: 开始步数偏移量
        """
        # 设置偏移量
        self.offset = offset
        
        # 确保只收集预期的步数
        expected_steps = self.offset + self.step_count
        
        # 运行模型步骤
        for i in tqdm(range(self.offset, self.step_count)):
            # 模型步进
            self.step()
            
            # 收集模型级数据
            self.datacollector.collect(self)
            
            # 检查点保存逻辑
            if checkpoint_path:
                # 每10步保存一次检查点
                if i % 10 == 0 and i > 0:
                    self.save_checkpoint(checkpoint_path + f"/{self.name}-{i}.pkl")
        
        # 检查数据收集器中的数据行数
        model_data = self.datacollector.get_model_vars_dataframe()
        if len(model_data) > expected_steps + 1:  # +1 for initial state
            print(f"WARNING: Too many data points collected: {len(model_data)}, expected {expected_steps + 1}")

    def save_checkpoint(self, file_path):
        '''保存检查点到指定文件路径'''
        with open(file_path, "wb") as file:
            pickle.dump(self, file)
    
    @staticmethod
    def load_checkpoint(file_path):
        '''从指定文件路径加载检查点'''
        with open(file_path, "rb") as file:
            return pickle.load(file)

    def check_consistency(self):
        """检查模型状态的一致性"""
        # 检查总人口
        total = self.susceptible + self.infected + self.recovered
        if total != self.population:
            print(f"ERROR: Population mismatch! {total} != {self.population}")
        
        # 检查每个代理人的状态与belief一致性
        for agent in self.schedule.agents:
            current_belief = agent.beliefs[-1] if agent.beliefs else None
            if (agent.health_condition == "Infected" and current_belief != 1) or \
               (agent.health_condition == "Susceptible" and current_belief != 0) or \
               (agent.health_condition == "Recovered" and current_belief != 0):
                print(f"WARNING: Agent {agent.unique_id} has inconsistent state: {agent.health_condition} with belief={current_belief}")
        
        # 检查感染计数
        infected_count = sum(1 for a in self.schedule.agents if a.health_condition == "Infected")
        susceptible_count = sum(1 for a in self.schedule.agents if a.health_condition == "Susceptible")
        recovered_count = sum(1 for a in self.schedule.agents if a.health_condition == "Recovered")
        
        if infected_count != self.infected:
            print(f"ERROR: Infected count mismatch! Actual: {infected_count}, Tracked: {self.infected}")
        
        if susceptible_count != self.susceptible:
            print(f"ERROR: Susceptible count mismatch! Actual: {susceptible_count}, Tracked: {self.susceptible}")
        
        if recovered_count != self.recovered:
            print(f"ERROR: Recovered count mismatch! Actual: {recovered_count}, Tracked: {self.recovered}")

    # 新增：保存对话数据
    def save_dialogue_data(self, file_path):
        """保存对话数据到JSON文件，确保包含完整对话内容"""
        # 检查对话记录是否为空
        if not self.dialogue_records:
            print("警告: 没有对话记录可保存!")
            # 创建一个空的对话数据结构
            dialogue_data = {
                "topic": self.topic,
                "population": self.population,
                "initial_healthy": self.initial_healthy,
                "initial_infected": self.initial_infected,
                "dialogues": [],
                "warning": "对话记录为空，请检查conduct_dialogue方法是否正确执行"
            }
        else:
            # 正常保存对话数据
            dialogue_data = {
                "topic": self.topic,
                "population": self.population,
                "initial_healthy": self.initial_healthy,
                "initial_infected": self.initial_infected,
                "dialogues": self.dialogue_records
            }
        
        # 调试输出
        print(f"保存对话数据: {len(self.dialogue_records)} 条对话记录")
        if self.dialogue_records:
            sample = self.dialogue_records[0]
            print(f"对话示例: 代理人 {sample.get('agents', '未知')} 的对话，{len(sample.get('history', []))} 轮")
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(dialogue_data, f, ensure_ascii=False, indent=2)
        
        print(f"对话数据已保存到 {file_path}")

    # 新增：保存代理人行为日志的方法
    def save_agent_behavior_logs(self, file_path):
        """保存所有代理人的行为日志到JSON文件"""
        behavior_data = {
            "simulation_info": {
                "topic": self.topic,
                "population": self.population,
                "initial_healthy": self.initial_healthy,
                "initial_infected": self.initial_infected,
                "total_steps": self.schedule.steps,
                "start_date": self.current_date.strftime("%Y-%m-%d")
            },
            "agents": []
        }
        
        for agent in self.schedule.agents:
            agent_data = {
                "id": agent.unique_id,
                "name": agent.name,
                "traits": agent.traits,
                "education": agent.qualification,
                "self_description": agent.self_description if hasattr(agent, "self_description") else "",
                "initial_health": "Infected" if agent.unique_id >= self.initial_healthy else "Susceptible",
                "final_health": agent.health_condition,
                "behavior_log": agent.behavior_log
            }
            behavior_data["agents"].append(agent_data)
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(behavior_data, f, ensure_ascii=False, indent=2)
        
        print(f"代理人行为日志已保存到 {file_path}")

    # 在update_day方法中添加健康状态变化记录
    def update_day(self, agent):
        '''更新代理人的健康状态'''
        old_health = agent.health_condition
        
        # 如果代理人相信谣言，则变为感染状态
        if agent.beliefs[-1] == 1 and agent.health_condition != "Infected":
            agent.health_condition = "Infected"
            agent.model.susceptible -= 1
            agent.model.infected += 1
            agent.model.daily_new_infected_cases += 1
            
            # 记录健康状态变化
            agent.log_behavior("健康状态变化", {
                "变化前": old_health,
                "变化后": "Infected",
                "原因": "相信谣言"
            })
        
        # 如果代理人不再相信谣言，则变为恢复状态
        elif agent.beliefs[-1] == 0 and agent.health_condition == "Infected":
            agent.health_condition = "Recovered"
            agent.model.infected -= 1
            agent.model.recovered += 1
            agent.model.daily_new_recovered_cases += 1
            
            # 记录健康状态变化
            agent.log_behavior("健康状态变化", {
                "变化前": old_health,
                "变化后": "Recovered",
                "原因": "不再相信谣言"
            })

    def monitor_memory_usage(self):
        """监控内存使用情况"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_usage_mb = memory_info.rss / 1024 / 1024
        print(f"当前内存使用: {memory_usage_mb:.2f} MB")
        return memory_usage_mb