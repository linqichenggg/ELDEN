import random
from zhipuai import ZhipuAI

class ZhipuLLM:
    def __init__(self, config):
        self.global_config = config
        self.config = config['LLM_config']
        
        self.client = ZhipuAI(api_key=self.config['API_KEY'])
        
        self.call_time = 0
    
    def parse_response(self, response):
        return {
            'run_id': response.id,
            'time_stamp': response.created,
            'result': response.choices[0].message.content
        }
    
    def run(self, message_list, temperature=1.0, penalty_score=0.0):
        response = self.client.chat.completions.create(
            model=self.config['model_name'],
            messages=message_list,
            temperature=temperature,
        )
        response = self.parse_response(response)
        self.call_time += 1
        
        return response
    
    def fast_run(self, query, temperature=0.95, penalty_score=1.0, exception_times=10):
        response = None
        for et in range(exception_times):
            try:
                response = self.run([{"role": "user", "content": query}], temperature, penalty_score)
                break
            except Exception as e:
                print('[%d] LLM inference fails: %s' % (et, e))
        if not response:
            return ' '
        
        return response['result']
    
    def __deepcopy__(self, memo):
        return None


# 与LLM交互的用户模拟方法 (作为函数提供，将通过run_simulation.py注册到RumorSimulation类)
def create_llm_prompt_for_knowledge_test(self, user_id):
    """创建用于知识测试的LLM提示"""
    user = self.users[user_id]
    
    # 为每个用户随机打乱问题顺序
    user_questions = self.knowledge_questions.copy()
    random.shuffle(user_questions)
    
    # 根据用户知识水平调整提示语气
    knowledge_level = user['health_knowledge']
    if knowledge_level < 30:
        knowledge_desc = "非常低的"
        behavior_guide = "你应该对大多数健康知识问题回答错误，可能会相信一些常见的健康谣言。"
    elif knowledge_level < 50:
        knowledge_desc = "较低的"
        behavior_guide = "你应该对一些健康知识问题回答错误，可能会相信一些健康谣言。"
    elif knowledge_level < 70:
        knowledge_desc = "中等的"
        behavior_guide = "你应该对一些健康知识问题回答正确，但对一些复杂问题可能回答错误。"
    elif knowledge_level < 90:
        knowledge_desc = "较高的"
        behavior_guide = "你应该对大多数健康知识问题回答正确，但偶尔可能有误解。"
    else:
        knowledge_desc = "非常高的"
        behavior_guide = "你应该对几乎所有健康知识问题回答正确，很少相信健康谣言。"
    
    prompt = f"""你现在要扮演一位具有{knowledge_desc}健康知识水平（{knowledge_level}/100分）的老年人，回答以下健康知识问题。
你的传播倾向为{user['transmission_tendency']:.2f}（0-1，越高越喜欢分享信息）。
你的谣言敏感度为{user['rumor_sensitivity']}（0-100，越高越容易相信谣言）。
你的社交圈大小为{user['social_circle_size']}（越大表示社交关系越多）。

{behavior_guide}

请根据上述特征，以这位老年人的角度回答下列问题。每个问题只需回答选项字母（A或B）。
记住：你的健康知识水平是{knowledge_level}/100分，这意味着你应该回答大约{knowledge_level}%的问题正确。

"""
    
    # 添加问题
    for i, q in enumerate(user_questions, 1):
        options_str = ""
        for j, opt in enumerate(q['options']):
            options_str += f"{chr(65+j)}. {opt} "
        
        prompt += f"问题{i}: {q['question']}\n{options_str}\n"
    
    # 明确告知LLM需要回答问题
    prompt += f"\n请直接回答上述所有问题，每个问题只需回答选项字母（如A或B）。请确保你的回答反映了你扮演的老年人的健康知识水平（{knowledge_level}/100分）。"
    
    # 添加用户ID作为唯一标识符
    prompt += f"\n\n用户ID: {user_id}"
    
    return prompt

def call_llm_for_knowledge_test(self, user_id):
    """调用LLM进行知识测试"""
    # 确保有知识问题
    if not self.knowledge_questions:
        self.setup_knowledge_questions()
    
    # 获取用户信息
    user = self.users[user_id]
    old_knowledge_level = user["health_knowledge"]
    
    # 记录问题
    self.log(f"用户{user_id}的知识测试问题:")
    for i, q in enumerate(self.knowledge_questions, 1):
        options_str = ", ".join([f"{chr(65+j)}. {opt}" for j, opt in enumerate(q['options'])])
        self.log(f"问题{i}: {q['question']} | 选项: {options_str} | 正确答案: {q['correct']}")
    
    try:
        if self.llm:
            # 创建提示 - 添加随机字符串以避免缓存
            prompt = self.create_llm_prompt_for_knowledge_test(user_id)
            # 添加随机字符串以确保每次提示都不同
            random_str = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=8))
            prompt += f"\n\n(请忽略此标识符: {random_str})"
            
            self.log(f"发送给LLM的提示: {prompt}")
            
            # 调用LLM - 增加temperature以增加随机性
            response = self.llm.fast_run(prompt, temperature=0.7)
            self.log(f"LLM原始回答: {response}")
            
            # 解析答案
            answers = []
            # 分割回答，处理可能的逗号分隔或换行分隔
            if ',' in response:
                parts = response.split(',')
            else:
                parts = response.split('\n')
            
            for part in parts:
                part = part.strip()
                if part and part[0] in 'ABab':
                    answers.append(part[0].upper())
            
            self.log(f"解析后的答案: {answers}")
            
            # 如果没有足够的答案，使用随机答案补充
            while len(answers) < len(self.knowledge_questions):
                # 根据用户知识水平调整随机答案的正确概率
                knowledge_factor = user["health_knowledge"] / 100
                correct_q = self.knowledge_questions[len(answers)]
                correct_letter = chr(65 + correct_q['options'].index(correct_q['correct']))
                
                if random.random() < knowledge_factor:
                    # 知识水平高的用户更可能答对
                    answers.append(correct_letter)
                else:
                    # 知识水平低的用户更可能答错
                    wrong_letter = 'B' if correct_letter == 'A' else 'A'
                    answers.append(wrong_letter)
            
            # 计算正确答案数量
            correct_count = 0
            for i, q in enumerate(self.knowledge_questions):
                if i < len(answers):
                    user_answer = answers[i]
                    correct_index = q['options'].index(q['correct'])
                    correct_letter = chr(65 + correct_index)
                    
                    if user_answer == correct_letter:
                        correct_count += 1
                        self.log(f"问题{i+1}回答正确: {user_answer} ({q['correct']})")
                    else:
                        self.log(f"问题{i+1}回答错误: {user_answer}，正确答案是 {correct_letter} ({q['correct']})")
            
            # 计算新的知识水平
            new_knowledge_level = int(correct_count / len(self.knowledge_questions) * 100)
            
            # 添加一些随机波动，避免所有用户得分相同
            new_knowledge_level += random.randint(-5, 5)
            new_knowledge_level = max(0, min(new_knowledge_level, 100))
            
            # 更新用户知识水平
            self.log(f"用户{user_id}知识测试结果: {correct_count}/{len(self.knowledge_questions)}题正确")
            self.log(f"知识水平从{old_knowledge_level}更新为{new_knowledge_level}")
            
            user["health_knowledge"] = new_knowledge_level
            user["rumor_sensitivity"] = 100 - new_knowledge_level
            user["knowledge_tested"] = True
            
            # 更新网络中的节点属性
            self.social_network.nodes[user_id]["health_knowledge"] = new_knowledge_level
            self.social_network.nodes[user_id]["rumor_sensitivity"] = 100 - new_knowledge_level
            
            return new_knowledge_level
            
        else:
            # 如果没有LLM，使用随机模拟
            return self.test_user_knowledge(user_id)
            
    except Exception as e:
        self.log(f"LLM知识测试失败: {str(e)}", level="ERROR")
        # 出错时使用随机模拟
        return self.test_user_knowledge(user_id)

def create_llm_prompt_for_message_decision(self, user_id, message):
    """创建用于消息决策的LLM提示"""
    user = self.users[user_id]
    sender = message.get("sender", "未知")
    
    prompt = f"""作为ID为{user_id}的老年人，具有以下特征：
- 健康知识水平：{user['health_knowledge']}/100
- 传播倾向：{user['transmission_tendency']*100}%
- 谣言敏感度：{user['rumor_sensitivity']}
- 社交圈大小：{user['social_circle_size']}人

你刚刚从朋友{sender}那里收到了这条信息："{message['content']}"

请回答以下问题：
1. 你对这条信息的第一反应是什么？(简短描述)
2. 你会选择验证这条信息的真实性吗？(只回答"是"或"否")
3. 如果选择验证，你会通过什么方式验证？(简短回答)
4. 你会将这条信息转发给你社交圈中的朋友吗？(只回答"是"或"否")

请基于你的角色特征做出真实反应，每个问题的回答都以数字编号开头。
"""
    return prompt

def call_llm_for_message_decision(self, user_id, message):
    """调用LLM做出关于消息的决策"""
    prompt = self.create_llm_prompt_for_message_decision(user_id, message)
    self.log(f"发送给LLM的消息决策提示: {prompt}")
    
    try:
        if self.llm:
            decision_text = self.llm.fast_run(prompt, temperature=0.7)
            self.log(f"用户{user_id}的消息决策LLM回答: {decision_text}")
            
        else:
            # 没有LLM时使用模拟决策
            return self.simulate_message_decision(user_id, message)
            
        # 解析决策
        decision_parts = decision_text.split("\n")
        verify = False
        forward = False
        
        for part in decision_parts:
            if "2." in part and "是" in part.lower():
                verify = True
            if "4." in part and "是" in part.lower():
                forward = True
        
        result = {
            "text": decision_text,
            "verify": verify,
            "forward": forward
        }
        
        self.log(f"决策结果: 验证={verify}, 转发={forward}")
        return result
        
    except Exception as e:
        self.log(f"LLM调用失败: {str(e)}", level="ERROR")
        # 模拟一个基于用户属性的随机决策
        return self.simulate_message_decision(user_id, message)

def simulate_message_decision(self, user_id, message):
    """当LLM不可用时模拟用户决策"""
    user = self.users[user_id]
    verify_prob = 1 - (user["rumor_sensitivity"] / 200)  # 健康知识越高越可能验证
    forward_prob = user["transmission_tendency"] * 1.2  # 传播倾向直接影响转发概率
    
    if message["type"] == "rumor":
        # 如果是谣言，降低转发概率
        forward_prob *= 0.8
        
    verify = random.random() < verify_prob
    forward = random.random() < forward_prob
    
    # 如果用户验证了谣言，就不太可能转发
    if verify and message["type"] == "rumor":
        forward = random.random() < 0.2
        
    result = {
        "text": "模拟决策",
        "verify": verify,
        "forward": forward
    }
    
    self.log(f"模拟决策结果: 验证={verify}, 转发={forward}")
    return result

def create_llm_prompt_for_debunk_reaction(self, user_id, debunk_message, original_rumor):
    """创建用于辟谣反应的LLM提示"""
    user = self.users[user_id]
    
    prompt = f"""作为ID为{user_id}的老年人，具有以下特征：
- 健康知识水平：{user['health_knowledge']}/100
- 传播倾向：{user['transmission_tendency']*100}%
- 谣言敏感度：{user['rumor_sensitivity']}
- 社交圈大小：{user['social_circle_size']}人

你最近看到了一条关于健康的官方辟谣信息："{debunk_message['content']}"

这条辟谣信息针对的是你之前可能看到过的信息："{original_rumor['content']}"

请回答以下问题：
1. 看到辟谣后，你的看法有变化吗？(只回答"是"或"否")
2. 如果你之前已经转发了原始信息，你会怎么做？(只回答：删除/发布更正/不做任何事)
3. 这次经历会影响你未来验证健康信息的行为吗？(简短回答)

请基于你的角色特征做出真实反应，每个问题的回答都以数字编号开头。
"""
    return prompt

def call_llm_for_debunk_reaction(self, user_id, debunk_message, original_rumor):
    """调用LLM模拟用户对辟谣信息的反应"""
    prompt = self.create_llm_prompt_for_debunk_reaction(user_id, debunk_message, original_rumor)
    self.log(f"发送给LLM的辟谣反应提示: {prompt}")
    
    try:
        if self.llm:
            reaction_text = self.llm.fast_run(prompt, temperature=0.7)
            self.log(f"用户{user_id}的辟谣反应LLM回答: {reaction_text}")
            
        else:
            # 没有LLM时使用模拟反应
            return self.simulate_debunk_reaction(user_id)
            
        # 解析反应
        reaction_parts = reaction_text.split("\n")
        changed_mind = False
        action = "不做任何事"
        
        for part in reaction_parts:
            if "1." in part and "是" in part.lower():
                changed_mind = True
            if "2." in part:
                if "删除" in part.lower():
                    action = "删除"
                elif "更正" in part.lower():
                    action = "发布更正"
        
        result = {
            "text": reaction_text,
            "changed_mind": changed_mind,
            "action": action
        }
        
        self.log(f"辟谣反应结果: 改变想法={changed_mind}, 行动={action}")
        return result
        
    except Exception as e:
        self.log(f"LLM调用失败: {str(e)}", level="ERROR")
        # 模拟一个基于用户属性的随机反应
        return self.simulate_debunk_reaction(user_id)

def simulate_debunk_reaction(self, user_id):
    """当LLM不可用时模拟用户对辟谣的反应"""
    user = self.users[user_id]
    
    # 健康知识越高，越可能改变想法
    change_prob = min(0.4 + user["health_knowledge"] / 200, 0.9)
    changed_mind = random.random() < change_prob
    
    # 决定行动
    action_probs = {
        "删除": 0.3 * user["health_knowledge"] / 100,
        "发布更正": 0.2 * user["health_knowledge"] / 100,
        "不做任何事": 1 - (0.5 * user["health_knowledge"] / 100)
    }
    
    actions = list(action_probs.keys())
    probabilities = list(action_probs.values())
    action = random.choices(actions, weights=probabilities, k=1)[0]
    
    result = {
        "text": "模拟辟谣反应",
        "changed_mind": changed_mind,
        "action": action
    }
    
    self.log(f"模拟辟谣反应结果: 改变想法={changed_mind}, 行动={action}")
    return result