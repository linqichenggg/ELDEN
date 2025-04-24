# -- coding: utf-8 --**
import mesa
from utils import get_completion_from_messages, get_completion_from_messages_json, format_dialogue_history
import logging
logger = logging.getLogger()
logger.setLevel(logging.WARNING)
import json
from prompt import *
import random

def get_summary_long(long_mem, short_mem):
    if not long_mem:
        return short_mem
    
    user_msg = f"""
    请将以下两段内容整合为一个连贯的摘要，保留关键信息：

    长期记忆：{long_mem}
    
    新信息：{short_mem}
    """
    
    msg = [{"role": "user", "content": user_msg}]
    response = get_completion_from_messages(msg, temperature=0.5)
    
    return response

def get_summary_short(opinions, topic):
    opinions_text = "\n".join(f"One people think: {opinion}" for opinion in opinions)
    user_msg = reflecting_prompt.format(opinions=opinions_text, topic=topic)
    msg = [{"role": "user", "content": user_msg}]
    return get_completion_from_messages(msg, temperature=0.5)

# 获取对话摘要
def get_dialogue_summary(dialogue_content, topic):
    user_msg = dialogue_summary_prompt.format(dialogue_content=dialogue_content, topic=topic)
    msg = [{"role": "user", "content": user_msg}]
    return get_completion_from_messages(msg, temperature=0.5)

class Citizen(mesa.Agent):
    '''
    Define who a citizen is:
    unique_id: assigns ID to agent
    name: name of the agent
    age: age of the agent (65-85)
    traits: big 5 traits of the agent
    health_condition: flag to say if Susceptible or Infected or Recovered
    '''

    def __init__(self, model, unique_id, name, age, traits, qualification, health_condition, opinion, topic, initial_memory=""):
        super().__init__(unique_id, model)  # 正确的参数顺序是 unique_id, model
        #Persona
        self.name = name
        self.age = age
        self.opinion = opinion
        self.traits = traits
        self.qualification = qualification
        self.topic = topic
        self.opinions = []
        self.beliefs = []
        self.long_opinion_memory = []  # 长期记忆应该是列表
        self.long_memory_full = []
        self.short_opinion_memory = []
        self.reasonings = []
        self.contact_ids = []
        # 对话历史和社交网络
        self.dialogue_history = []  # 存储所有对话
        self.interaction_history = []  # 存储互动过的代理人ID
        self.dialogue_partners = {}  # 存储与每个代理人的对话次数
        self.dialogue_summaries = {}  # 存储与每个代理人的对话摘要
        self.self_description = ""  # 初始化自我描述

        #Health Initialization of Agent
        self.health_condition = health_condition

        #Contact Rate  
        self.agent_interaction = []

        #Reasoning tracking
        self.initial_belief = 1 if health_condition == 'Infected' else 0
        self.initial_reasoning = 'initial_reasoning'
        self.opinions.append(self.opinion)
        self.beliefs.append(self.initial_belief)
        self.reasonings.append(self.initial_reasoning)

        # 添加行为记录列表
        self.behavior_log = []
        
        # 记录初始状态
        self.log_behavior("初始化", {
            "信念": 1 if health_condition == "Infected" else 0,
            "健康状态": health_condition,
            "观点": opinion,
            "日期": model.current_date.strftime("%Y-%m-%d")
        })

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
        '''与其他代理人互动并更新观点 - 单轮对话版本，保留用于兼容'''
        # 收集其他人的观点
        others_opinions = [agent.opinions[-1] for agent in self.agent_interaction]
        
        # 生成观点摘要
        opinion_short_summary = get_summary_short(others_opinions, topic=self.topic)
        
        # 更新长期记忆
        long_mem = get_summary_long(self.long_opinion_memory, opinion_short_summary)
        
        # 构建提示信息
        user_msg = update_opinion_prompt.format(
            agent_name=self.name,
            openness=self.traits["开放性"],
            conscientiousness=self.traits["尽责性"],
            extraversion=self.traits["外向性"],
            agreeableness=self.traits["宜人性"],
            neuroticism=self.traits["神经质"],
            agent_qualification=self.qualification,
            self_description=self.self_description,
            topic=self.topic,
            opinion="【重要】" + self.opinion,
            long_mem=long_mem,
            others_opinions=opinion_short_summary
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
    
    # 生成对话初始响应
    def generate_dialogue_initiation(self, other_agent):
        """生成对话的开场白"""
        user_msg = dialogue_initiation_prompt.format(
            agent_name=self.name,
            openness=self.traits["开放性"],
            conscientiousness=self.traits["尽责性"],
            extraversion=self.traits["外向性"],
            agreeableness=self.traits["宜人性"],
            neuroticism=self.traits["神经质"],
            agent_qualification=self.qualification,
            self_description=self.self_description,
            topic=self.topic,
            current_opinion=self.opinions[-1],
            other_name=other_agent.name
        )
        
        msg = [{"role": "user", "content": user_msg}]
        response_json = get_completion_from_messages_json(msg, temperature=0.7)
        
        try:
            response_data = json.loads(response_json)
            # 记录对话发起
            self.log_behavior("对话发起", {
                "对话伙伴ID": other_agent.unique_id,
                "对话伙伴名称": other_agent.name,
                "对话内容": response_data["response"] if "response" in response_data else "",
                "对话主题": self.topic
            })
            return response_data
        except:
            # 默认响应
            return {
                "response": f"我是{self.name}，我想和您讨论一下关于{self.topic}的看法。",
                "internal_thoughts": "我将表达我的观点",
                "belief_shift": 0,
                "reasoning": "这是对话的开始"
            }
    
    # 生成对话响应
    def generate_dialogue_response(self, conversation_history, dialogue_state, other_agent):
        """生成对话中的响应"""
        # 获取当前轮次
        turn_number = int(dialogue_state.turn_count) + 1
        
        # 获取对方最后的回应
        if conversation_history:
            other_response = conversation_history[-1]["content"]
        else:
            other_response = "（对话开始）"
        
        # 格式化对话历史
        formatted_history = format_dialogue_history(conversation_history)
        
        # 构建提示信息
        user_msg = multi_turn_dialogue_prompt.format(
            agent_name=self.name,
            openness=self.traits["开放性"],
            conscientiousness=self.traits["尽责性"],
            extraversion=self.traits["外向性"],
            agreeableness=self.traits["宜人性"],
            neuroticism=self.traits["神经质"],
            agent_qualification=self.qualification,
            self_description=self.self_description,
            topic=self.topic,
            current_opinion=self.opinions[-1],
            other_name=other_agent.name,
            turn_number=turn_number,
            conversation_history=formatted_history,
            other_response=other_response
        )
        
        msg = [{"role": "user", "content": user_msg}]
        response_json = get_completion_from_messages_json(msg, temperature=0.7)
        
        try:
            response_data = json.loads(response_json)
            # 记录对话回应
            self.log_behavior("对话回应", {
                "对话伙伴ID": other_agent.unique_id,
                "对话伙伴名称": other_agent.name,
                "对话内容": response_data["response"] if "response" in response_data else "",
                "对话轮次": dialogue_state.turn_count,
                "共同点程度": dialogue_state.common_ground
            })
            return response_data
        except:
            # 默认响应
            return {
                "response": f"我理解您的观点，但我需要再考虑一下。",
                "internal_thoughts": "我无法完全理解对方的观点",
                "belief_shift": 0,
                "reasoning": "需要更多信息才能做出判断"
            }
    
    # 更新对话后的信念
    def update_belief_after_dialogue(self, belief_change, conversation_history):
        """对话结束后更新信念"""
        # 获取当前信念
        current_belief = self.beliefs[-1]
        
        # 计算新信念值
        new_belief_value = current_belief
        
        # 如果信念变化足够大，则改变信念
        if abs(belief_change) >= 0.3:
            if current_belief == 1 and belief_change < 0:
                new_belief_value = 0
            elif current_belief == 0 and belief_change > 0:
                new_belief_value = 1
        
        # 生成新的观点和推理
        if new_belief_value != current_belief:
            # 信念发生变化，生成新的观点
            if new_belief_value == 1:
                new_opinion = f"我现在相信：{random.choice(topic_to_sentences[self.topic]['infeted'])}"
            else:
                new_opinion = f"我现在不相信：{random.choice(topic_to_sentences[self.topic]['susceptible'])}"
            
            # 从对话中提取推理
            new_reasoning = "通过对话改变了看法"
            for turn in conversation_history:
                if turn["speaker"] == self.name and "reasoning" in turn:
                    new_reasoning = turn["reasoning"]
                    break
        else:
            # 信念未变，保持原有观点但可能有细微调整
            if current_belief == 1:
                new_opinion = f"我依然相信：{random.choice(topic_to_sentences[self.topic]['infeted'])}"
            else:
                new_opinion = f"我依然不相信：{random.choice(topic_to_sentences[self.topic]['susceptible'])}"
            
            new_reasoning = self.reasonings[-1]
        
        # 更新信念、观点和推理
        self.beliefs.append(new_belief_value)
        self.opinions.append(new_opinion)
        self.reasonings.append(new_reasoning)
        
        # 打印更新信息
        print(f"Agent {self.unique_id} 信念更新: {current_belief} -> {new_belief_value}")
        print(f"新观点: {new_opinion}")
        print(f"推理: {new_reasoning}")
        
        # 更新对话历史记录
        dialogue_content = format_dialogue_history(conversation_history)
        dialogue_summary = get_dialogue_summary(dialogue_content, self.topic)
        
        # 更新长期记忆
        self.update_long_memory_with_dialogue(dialogue_summary)
        
        # 记录信念变化
        self.log_behavior("信念更新", {
            "对话前信念": current_belief,
            "对话后信念": new_belief_value,
            "信念变化量": belief_change,
            "对话伙伴": conversation_history[0]['speaker'] if conversation_history else "未知",
            "最终观点": new_opinion if new_opinion else "",
            "健康状态": self.health_condition
        })
        
        return new_belief_value
    
    # 用对话更新长期记忆
    def update_long_memory_with_dialogue(self, dialogue_summary):
        """将对话摘要整合到长期记忆中"""
        # 如果长期记忆为空，直接使用对话摘要
        if not self.long_opinion_memory:
            self.long_opinion_memory = dialogue_summary
            return
        
        # 否则，整合对话摘要到长期记忆
        updated_memory = get_summary_long(self.long_opinion_memory, dialogue_summary)
        self.long_opinion_memory = updated_memory
        self.long_memory_full.append(updated_memory)

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
        # 在多轮对话模式下，这个函数不再直接调用interact
        # 而是由World类的conduct_dialogue方法管理对话过程
        pass

    # 更新长期记忆
    def update_long_memory(self):
        """更新长期记忆"""
        if not self.long_opinion_memory:
            return
        
        user_msg = long_memory_prompt.format(
            agent_name=self.name,
            openness=self.traits["开放性"],
            conscientiousness=self.traits["尽责性"],
            extraversion=self.traits["外向性"],
            agreeableness=self.traits["宜人性"],
            neuroticism=self.traits["神经质"],
            agent_qualification=self.qualification,
            self_description=self.self_description,
            topic=self.topic,
            long_mem=self.long_opinion_memory
        )
        
        msg = [{"role": "user", "content": user_msg}]
        response = get_completion_from_messages(msg, temperature=0.7)
        
        self.long_opinion_memory = response
        self.long_memory_full.append(response)

    # 反思社区观点
    def reflect_on_community(self, community_opinions):
        """反思社区观点并可能更新信念"""
        user_msg = reflecting_prompt.format(
            agent_name=self.name,
            openness=self.traits["开放性"],
            conscientiousness=self.traits["尽责性"],
            extraversion=self.traits["外向性"],
            agreeableness=self.traits["宜人性"],
            neuroticism=self.traits["神经质"],
            agent_qualification=self.qualification,
            self_description=self.self_description,
            topic=self.topic,
            opinion=self.opinions[-1],
            long_mem=self.long_opinion_memory,
            community_opinions=community_opinions
        )
        
        msg = [{"role": "user", "content": user_msg}]
        response_json = get_completion_from_messages_json(msg, temperature=0.7)
        
        try:
            output = json.loads(response_json)
            reflection = output.get('reflection', "无法形成反思")
            updated_belief = int(output.get('updated_belief', self.beliefs[-1]))
            reasoning = output.get('reasoning', "无法解释原因")
            
            # 如果信念发生变化，更新观点
            if updated_belief != self.beliefs[-1]:
                if updated_belief == 1:
                    new_opinion = f"经过反思，我相信：{random.choice(topic_to_sentences[self.topic]['infeted'])}"
                else:
                    new_opinion = f"经过反思，我不相信：{random.choice(topic_to_sentences[self.topic]['susceptible'])}"
                
                self.opinions.append(new_opinion)
                self.beliefs.append(updated_belief)
                self.reasonings.append(reasoning)
                
                print(f"Agent {self.unique_id} 通过反思更新了信念: {self.beliefs[-2]} -> {updated_belief}")
                print(f"反思: {reflection}")
                print(f"新观点: {new_opinion}")
                print(f"推理: {reasoning}")
            
            return reflection, updated_belief, reasoning
        except:
            print(f"Agent {self.unique_id} 反思处理失败")
            return "反思处理失败", self.beliefs[-1], "处理错误"

    def get_long_term_memory(self):
        '''获取长期记忆内容'''
        if not self.long_opinion_memory:
            return ""
        elif isinstance(self.long_opinion_memory, str):
            return self.long_opinion_memory
        else:
            # 将列表中的记忆合并为字符串
            return "\n\n".join(self.long_opinion_memory)

    # 添加记录行为的方法
    def log_behavior(self, action_type, details):
        """记录代理人的行为
        
        参数:
            action_type: 行为类型 (如 "对话", "信念更新", "健康状态变化" 等)
            details: 包含行为详情的字典
        """
        timestamp = self.model.current_date.strftime("%Y-%m-%d") + f" (步骤 {self.model.schedule.steps})"
        
        log_entry = {
            "时间戳": timestamp,
            "代理人ID": self.unique_id,
            "代理人姓名": self.name,
            "行为类型": action_type,
            "细节": details
        }
        
        self.behavior_log.append(log_entry)