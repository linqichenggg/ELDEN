import mesa
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import random
import pickle
from citizen import Citizen
from tqdm import tqdm
from datetime import datetime, timedelta
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
        self.track_contact_rate = [0]
        self.list_new_infected_cases = [0]
        self.list_new_susceptible_cases = [0]
        self.daily_new_infected_cases = initial_infected
        self.daily_new_susceptible_cases = initial_healthy
        self.infected = initial_infected
        self.susceptible = initial_healthy
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
        for agent in self.schedule.agents:
            self.total_contact_rates += len(agent.agent_interaction)
        self.track_contact_rate.append(self.total_contact_rates)
        self.total_contact_rates = 0

        # 调用每个代理人的step函数
        self.schedule.step()

        # 确保在这里调用update_day
        print("Updating agent days...")
        for agent in self.schedule.agents:
            update_day(agent)
            print(f"Agent {agent.unique_id}: {agent.health_condition} (belief={agent.belief})")

        # 在world.py的step方法结束时
        belief_count = sum(a.belief for a in self.schedule.agents)
        infected_count = sum(1 for a in self.schedule.agents if a.health_condition == "Infected")
        if belief_count != infected_count:
            print(f"WARNING: Belief count ({belief_count}) does not match infected count ({infected_count})")

    def run_model(self, checkpoint_path, offset=0):
        '''运行模型'''
        self.offset = offset
        end_program = 0
        
        for i in tqdm(range(self.offset, self.step_count)):
            # 收集模型级数据
            self.datacollector.collect(self)

            # 模型步进
            self.step()  

            # 收集一天中的所有新病例
            self.list_new_infected_cases.append(self.daily_new_infected_cases)
            self.list_new_susceptible_cases.append(self.daily_new_susceptible_cases)
            
            # 打印语句
            print(f"Total Population: {self.population}")
            print(f"Currently Infected: {self.infected} | Currently Susceptible: {self.susceptible}")
            print(f"New Infections: {self.daily_new_infected_cases} | New Recoveries: {self.daily_new_susceptible_cases}")
            
            # 重置每日新病例
            self.daily_new_infected_cases = 0
            self.daily_new_susceptible_cases = 0

            # 早期停止条件
            if self.infected == 0:
                end_program += 1
            if end_program == 2:
                path = checkpoint_path + f"/{self.name}-final_early.pkl"
                self.save_checkpoint(file_path=path)
                break

            # 更新日期并保存检查点
            self.current_date += timedelta(days=1)
            path = checkpoint_path + f"/{self.name}-{i+1}.pkl"
            self.save_checkpoint(file_path=path)
            clear_cache()

    def save_checkpoint(self, file_path):
        '''保存检查点到指定文件路径'''
        with open(file_path, "wb") as file:
            pickle.dump(self, file)
    
    @staticmethod
    def load_checkpoint(file_path):
        '''从指定文件路径加载检查点'''
        with open(file_path, "rb") as file:
            return pickle.load(file)
