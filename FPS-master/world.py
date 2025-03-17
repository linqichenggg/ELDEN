import mesa
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import random
import pickle
from citizen import Citizen
from tqdm import tqdm
from datetime import datetime
from utils import generate_names, generate_big5_traits, generate_qualifications, update_day, clear_cache
from prompt import *

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
        
        # 删除社交场景相关内容

        # 感染相关变量
        self.total_contact_rates = 0
        self.daily_contact_count = 0  # 添加这个变量
        
        # 初始化列表并添加初始值
        self.track_contact_rate = [0]  # 初始步骤的接触率为0
        self.list_new_infected_cases = [self.initial_infected]  # 初始感染数
        self.list_new_susceptible_cases = [0]
        self.list_new_recovered_cases = [0]  # 初始没有恢复
        
        # 每日计数器
        self.daily_new_infected_cases = 0
        self.daily_new_susceptible_cases = 0
        self.daily_new_recovered_cases = 0
        
        self.infected = initial_infected
        self.susceptible = initial_healthy
        self.recovered = 0
        
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
        
        # 生成代理人属性
        names = generate_names(self.population, self.population*2)
        traits = generate_big5_traits(self.population)
        qualifications = generate_qualifications(self.population)

        # 初始化代理人
        for i in range(self.population):
            agent_id = i  # 简化ID分配
            
            # 创建健康或感染的代理人
            if i < self.initial_healthy:
                health_condition = "Susceptible"
                opinion = random.choice(topic_sentence_susceptible)
                # 添加更明确的不信标记
                opinion = "我不相信：" + opinion
            else:
                health_condition = "Infected"
                opinion = random.choice(topic_sentence_infeted)
                # 添加更明确的相信标记
                opinion = "我相信：" + opinion

            # 创建Citizen实例
            citizen = Citizen(
                model=self,
                unique_id=agent_id, 
                name=names[i], 
                age=random.randrange(60, 90),
                traits=traits[i], 
                opinion=opinion,
                qualification=qualifications[i],
                health_condition=health_condition,
                topic=self.topic
            )  
            
            # 添加代理人到调度器
            self.schedule.add(citizen)

        # 在初始化代理人后，手动记录初始状态
        self.datacollector.collect(self)  # 收集初始状态
        
        # 初始化完成后检查一致性
        self.check_consistency()
        
        # 打印调试信息
        print(f"初始化完成: track_contact_rate={len(self.track_contact_rate)}, list_new_infected_cases={len(self.list_new_infected_cases)}")

    def decide_agent_interactions(self):
        '''决定代理人之间的互动'''
        # 基本互动
        for agent in self.schedule.agents:
            potential_interactions = [a for a in self.schedule.agents if a is not agent]  
            random.shuffle(potential_interactions) 
            potential_interactions = potential_interactions[:self.contact_rate]  
            for other_agent in potential_interactions:
                agent.agent_interaction.append(other_agent)    

        # 删除老年人特色社交代码

    def step(self):
        '''模型时间步进'''
        # 决定代理人互动
        self.decide_agent_interactions()  
       
        # 跟踪全局接触率
        self.daily_contact_count = 0  # 重置每日接触计数
        for agent in self.schedule.agents:
            self.daily_contact_count += len(agent.agent_interaction)
        
        # 调用每个代理人的step函数
        self.schedule.step()

        # 确保在这里调用update_day
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

        # 在步骤结束时记录接触率
        self.track_contact_rate.append(self.daily_contact_count)
        print(f"步骤结束: track_contact_rate={len(self.track_contact_rate)}, daily_contact_count={self.daily_contact_count}")

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
            
            # 检查点保存逻辑...
        
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

    def collect_data(self, step):
        """收集当前步骤的数据"""
        # 收集模型级数据
        self.datacollector.collect(self)
        
        # 确保列表长度一致
        expected_length = step + 1  # 包括初始状态
        
        # 检查并调整接触率列表
        while len(self.track_contact_rate) < expected_length:
            self.track_contact_rate.append(0)
        
        # 检查并调整新增感染列表
        while len(self.list_new_infected_cases) < expected_length:
            self.list_new_infected_cases.append(0)
        
        # 检查并调整新增恢复列表
        while len(self.list_new_recovered_cases) < expected_length:
            self.list_new_recovered_cases.append(0)
        
        # 记录当前步骤的数据
        if step > 0:  # 不是初始状态
            self.list_new_infected_cases[step] = self.daily_new_infected_cases
            self.list_new_recovered_cases[step] = self.daily_new_recovered_cases
            self.track_contact_rate[step] = self.daily_contact_count
        
        # 重置每日计数器
        self.daily_new_infected_cases = 0
        self.daily_new_susceptible_cases = 0
        self.daily_new_recovered_cases = 0
        self.daily_contact_count = 0
