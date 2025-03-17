# -- coding: utf-8 --**
import mesa
from utils import get_completion_from_messages, get_completion_from_messages_json
import logging
logger = logging.getLogger()
logger.setLevel(logging.WARNING)
import json
from prompt import *
import random

def get_summary_long(long_memory, short_memory):
    user_msg = long_memory_prompt.format(long_memory=long_memory, short_memory=short_memory)
    msg = [{"role": "user", "content": user_msg}]
    return get_completion_from_messages(msg, temperature=1)

def get_summary_short(opinions, topic):
    opinions_text = "\n".join(f"One people think: {opinion}" for opinion in opinions)
    user_msg = reflecting_prompt.format(opinions=opinions_text, topic=topic)
    msg = [{"role": "user", "content": user_msg}]
    return get_completion_from_messages(msg, temperature=0.5)

class Citizen(mesa.Agent):
    '''
    Define who a citizen is:
    unique_id: assigns ID to agent
    name: name of the agent
    age: age of the agent
    traits: big 5 traits of the agent
    health_condition: flag to say if Susceptible or Infected or Recovered
    day_infected: agent attribute to count the number of days agent spends infected
    width, height: dimensions of world
    '''

    def __init__(self, model, unique_id, name, age, traits, qualification, health_condition, opinion, topic):
        super().__init__(unique_id, model)  # 正确的参数顺序是 unique_id, model
        #Persona
        self.name = name
        self.age = age
        self.opinion=opinion
        self.traits=traits
        self.qualification=qualification
        self.topic=topic
        self.opinions = []
        self.beliefs = []
        self.long_opinion_memory = ''
        self.long_memory_full = []
        self.short_opinion_memory = []
        self.reasonings = []
        self.contact_ids = []

        #Health Initialization of Agent
        self.health_condition=health_condition

        #Contact Rate  
        self.agent_interaction=[]

        #Reasoning tracking
        self.persona = {"name":name, "age":age, "traits":traits}

        self.initial_belief = 1 if health_condition == 'Infected' else 0
        self.initial_reasoning = 'initial_reasoning'
        self.opinions.append(self.opinion)
        self.beliefs.append(self.initial_belief)
        self.reasonings.append(self.initial_reasoning)

        # 媒体使用习惯
        self.media_usage = {
            'traditional': random.choice(["每天看电视", "经常听广播", "读报纸"]),
            'digital': random.choice(["会用微信", "不会用智能手机", "子女帮忙操作"])
        }

    ########################################
    #          Initial Opinion             #
    ########################################
    def initial_opinion_belief(self):
        if self.health_condition == 'Infected':
            belief = 1
        else:  # Susceptible
            belief = 0

        reasoning = 'initial_reasoning'

        return belief, reasoning


    ################################################################################
    #                       Meet_interact_infect functions                         #
    ################################################################################ 

    def interact(self):
        '''与其他代理人互动并更新观点'''
        # 收集其他人的观点
        others_opinions = [agent.opinions[-1] for agent in self.agent_interaction]
        
        # 生成观点摘要
        opinion_short_summary = get_summary_short(others_opinions, topic=self.topic)
        
        # 更新长期记忆
        long_mem = get_summary_long(self.long_opinion_memory, opinion_short_summary)
        
        # 构建提示信息
        user_msg = update_opinion_prompt.format(
            agent_name=self.name,
            agent_age=self.age,
            agent_persona=self.traits,
            agent_qualification=self.qualification,
            media_usage=self.media_usage,
            topic=self.topic,
            opinion="【重要】" + self.opinion,
            long_mem=long_mem
        )
        
        # 获取新观点和信念
        self.opinion, self.belief, self.reasoning = self.response_and_belief(user_msg)
        self.opinions.append(self.opinion)
        self.beliefs.append(self.belief)
        self.reasonings.append(self.reasoning)
        
        # 打印结果
        print(f"ID: {self.unique_id}")
        print(f"Tweet: {self.opinion}")
        print(f"Belief: {self.belief}")
        print(f"Reasoning: {self.reasoning}")
        print("-" * 50)
        
        # 更新记忆
        self.long_opinion_memory = long_mem
        
        # 重置互动列表
        self.agent_interaction = []

    ########################################
    #               Infect                 #
    ########################################
        
    def response_and_belief(self, user_msg):
        '''获取LLM响应并提取信念'''
        msg = [{"role": "user", "content": user_msg}]
        response_json = get_completion_from_messages_json(msg, temperature=1)
        try:
            output = json.loads(response_json)
            tweet = output['tweet']
            belief = int(output['belief'])
            reasoning = output['reasoning']
            return tweet, belief, reasoning
        except:
            # 默认返回
            return "无法解析响应", 0, "处理错误"


    ################################################################################
    #                              step functions                                  #
    ################################################################################
  

    def step(self):
        '''代理人步进函数'''
        self.interact()
