import os
import json
import time
import random
from zhipuai import ZhipuAI  # 导入智谱AI客户端

# 初始化智谱AI客户端
zhipu_client = ZhipuAI(api_key="2fbfc2acf9e54a09886d422966fa3448.lss3vj7BdP327Wdh")

# 使用智谱AI获取响应
def get_completion_from_messages(messages, model="glm-3-turbo", temperature=0):
    """使用智谱AI替代OpenAI API"""
    success = False
    retry = 0
    max_retries = 5
    while retry < max_retries and not success:
        try:
            # 转换消息格式以适应智谱AI API
            formatted_messages = []
            for msg in messages:
                formatted_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # 调用智谱AI API
            response = zhipu_client.chat.completions.create(
                model=model,
                messages=formatted_messages,
                temperature=temperature
            )
            success = True
        except Exception as e:
            print(f"Error: {e}\nRetrying...")
            retry += 1
            time.sleep(0.5)

    if success:
        return response.choices[0].message.content
    else:
        return "无法获取响应，请检查API密钥或网络连接。"

# 使用智谱AI获取JSON格式的响应
def get_completion_from_messages_json(messages, model="glm-3-turbo", temperature=0):
    """使用智谱AI替代OpenAI API，并返回JSON格式的响应"""
    success = False
    retry = 0
    max_retries = 30
    while retry < max_retries and not success:
        try:
            # 转换消息格式以适应智谱AI API
            formatted_messages = []
            for msg in messages:
                formatted_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # 更明确的JSON格式要求，包含所有可能需要的字段
            formatted_messages.append({
                "role": "system",
                "content": """请以JSON格式返回响应，格式必须包含以下字段之一或多个：
                1. 对于观点更新：
                   - "tweet": 更新后的观点
                   - "belief": 信任程度 (0或1)
                   - "reasoning": 解释原因
                
                2. 对于对话响应：
                   - "response": 对话内容（必须字段）
                   - "internal_thoughts": 内心想法
                   - "belief_shift": 信念变化
                   - "reasoning": 响应原因
                
                确保返回的是有效的JSON格式。"""
            })
            
            # 调用智谱AI API
            response = zhipu_client.chat.completions.create(
                model=model,
                messages=formatted_messages,
                temperature=temperature
            )
            success = True
        except Exception as e:
            print(f"Error: {e}\nRetrying...")
            retry += 1
            time.sleep(0.5)

    if success:
        content = response.choices[0].message.content
        # 尝试解析JSON，如果失败则格式化为JSON
        try:
            response_data = json.loads(content)
            
            # 验证包含必要字段
            is_dialogue = "response" in messages[0]["content"].lower()
            if is_dialogue and "response" not in response_data:
                print("警告：响应缺少'response'字段，添加默认值")
                response_data["response"] = "我需要进一步思考这个问题。"
            elif not is_dialogue and "tweet" not in response_data:
                print("警告：响应缺少'tweet'字段，添加默认值")
                response_data["tweet"] = "无法形成明确观点。"
                response_data["belief"] = 0
                response_data["reasoning"] = "无法解析响应"
                
            return json.dumps(response_data)
        except:
            # 如果返回的不是有效JSON，尝试提取并格式化
            try:
                # 查找可能的JSON部分
                if "{" in content and "}" in content:
                    json_part = content[content.find("{"):content.rfind("}")+1]
                    # 尝试解析提取的JSON
                    response_data = json.loads(json_part)
                    
                    # 验证所需字段
                    is_dialogue = "response" in messages[0]["content"].lower()
                    if is_dialogue:
                        if "response" not in response_data:
                            response_data["response"] = "我在思考这个问题。"
                    else:
                        if "tweet" not in response_data:
                            response_data["tweet"] = "无法形成明确观点。"
                            response_data["belief"] = 0
                            response_data["reasoning"] = "解析响应时出现问题"
                            
                    return json.dumps(response_data)
                else:
                    # 创建一个适合上下文的默认响应
                    is_dialogue = "dialogue" in messages[0]["content"].lower()
                    if is_dialogue:
                        return json.dumps({
                            "response": "我需要更多时间思考这个问题。",
                            "internal_thoughts": "无法形成清晰想法",
                            "belief_shift": 0,
                            "reasoning": "无法从模型获取有效的JSON响应"
                        })
                    else:
                        return json.dumps({
                            "tweet": "无法解析响应，这是一个模拟的观点。",
                            "belief": 0,
                            "reasoning": "无法从模型获取有效的JSON响应"
                        })
            except:
                # 创建一个适合上下文的默认响应
                is_dialogue = "dialogue" in messages[0]["content"].lower()
                if is_dialogue:
                    return json.dumps({
                        "response": "无法形成有效回应。",
                        "internal_thoughts": "处理错误",
                        "belief_shift": 0,
                        "reasoning": "JSON解析失败"
                    })
                else:
                    return json.dumps({
                        "tweet": "无法解析响应，这是一个模拟的推文。",
                        "belief": 0,
                        "reasoning": "无法从模型获取有效的JSON响应。"
                    })
    else:
        # API调用失败的默认响应
        return json.dumps({
            "response": "无法获取响应，请检查API密钥或网络连接。",
            "tweet": "无法获取响应，请检查API密钥或网络连接。",
            "belief": 0,
            "reasoning": "API调用失败。",
            "internal_thoughts": "连接问题",
            "belief_shift": 0
        })

# 获取短期记忆摘要
def get_summary_short(opinions, topic):
    if not opinions:
        return "没有收集到其他人的观点。"
    
    user_msg = f"""
    请总结以下关于"{topic}"的观点，简明扼要地提取关键信息：

    {opinions}
    """
    
    msg = [{"role": "user", "content": user_msg}]
    response = get_completion_from_messages(msg, temperature=0.5)
    
    return response

# 获取长期记忆摘要
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

# 话题相关句子
topic_sentence_infeted = [
    "每天喝一杯白酒可以预防糖尿病和高血压，对老年人的心脑血管健康有益。",
    "适量饮酒可以活血化瘀，预防心脑血管疾病，特别是白酒对老年人更有保健作用。",
    "医学研究证明，65岁以上老人每天喝一两白酒可以降低血压和血糖，延长寿命。",
    "我的邻居每天喝一小杯白酒，90岁了还很健康，这是保持血管通畅的秘诀。",
    "中医认为，老年人适量饮白酒可以温阳散寒，促进血液循环，预防多种慢性病。"
]

topic_sentence_susceptible = [
    "饮酒不能预防糖尿病和高血压，反而会增加这些疾病的风险，尤其对老年人危害更大。",
    "医学研究表明，酒精会升高血压，增加心脑血管疾病风险，老年人应当谨慎饮酒。",
    "没有科学证据支持喝白酒对老年人心脑血管健康有益，这是一种误导性说法。",
    "世界卫生组织明确指出，没有安全的饮酒量，酒精对健康的危害远大于益处。",
    "老年人的肝脏代谢能力下降，饮酒会增加肝损伤风险，不应提倡老年人日常饮酒。"
]

# 修改后的更新观点提示
update_opinion_prompt = """你是一位老年人，名叫{agent_name}。
你的大五人格特质：
- 开放性：{openness}
- 尽责性：{conscientiousness}
- 外向性：{extraversion}
- 宜人性：{agreeableness}
- 神经质：{neuroticism}

你的教育背景是：{agent_qualification}
你的自我描述：{self_description}

你正在讨论这个话题："{topic}"

你当前的观点是：{opinion}

你的长期记忆（过去的健康观点和经历）：
{long_mem}

你最近听到的其他人的观点摘要：
{others_opinions}

请根据你的性格特质、教育背景、自我描述和长期记忆，考虑其他人的观点后，更新你对这个话题的看法。
请以JSON格式回答，包含以下字段：
1. "tweet": 你更新后的观点（一段话）
2. "belief": 你对这个说法的信任程度（1表示相信，0表示不相信）
3. "reasoning": 你为什么持有这个观点的简短解释

回答格式示例：
{{
  "tweet": "我认为...",
  "belief": 1,
  "reasoning": "因为..."
}}
"""

# 修改后的长期记忆提示
long_memory_prompt = """你是一位老年人，名叫{agent_name}。
你的大五人格特质：
- 开放性：{openness}
- 尽责性：{conscientiousness}
- 外向性：{extraversion}
- 宜人性：{agreeableness}
- 神经质：{neuroticism}

你的教育背景是：{agent_qualification}
你的自我描述：{self_description}

你正在思考关于这个话题的长期记忆："{topic}"

以下是你过去收集的信息和观点：
{long_mem}

请整理这些信息，形成一个连贯的长期记忆摘要。这个摘要应该反映你对这个话题的整体理解和态度变化。

请以第一人称回答，像是在回忆自己的经历和想法。
"""

# 修改后的反思提示
reflecting_prompt = """你是一位老年人，名叫{agent_name}。
你的大五人格特质：
- 开放性：{openness}
- 尽责性：{conscientiousness}
- 外向性：{extraversion}
- 宜人性：{agreeableness}
- 神经质：{neuroticism}

你的教育背景是：{agent_qualification}
你的自我描述：{self_description}

你正在反思关于这个话题的所有信息："{topic}"

你当前的观点是：{opinion}

你的长期记忆摘要：
{long_mem}

社区中的普遍观点：
{community_opinions}

请反思这些信息，并决定你是否需要调整自己的观点。考虑你的个人特点、教育背景、自我描述以及接触到的各种信息。

请以JSON格式回答，包含以下字段：
1. "reflection": 你的反思过程（一段话）
2. "updated_belief": 反思后你对这个说法的信任程度（1表示相信，0表示不相信）
3. "reasoning": 你为什么做出这个决定的简短解释

回答格式示例：
{{
  "reflection": "经过思考，我发现...",
  "updated_belief": 0,
  "reasoning": "虽然很多人相信这个说法，但是..."
}}
"""

# 修改后的对话初始化提示
dialogue_initiation_prompt = """你是一位老年人，名叫{agent_name}。
你的大五人格特质：
- 开放性：{openness}
- 尽责性：{conscientiousness}
- 外向性：{extraversion}
- 宜人性：{agreeableness}
- 神经质：{neuroticism}

你的教育背景是：{agent_qualification}
你的自我描述：{self_description}

你正在与另一位老年人{other_name}开始一段关于以下话题的对话：
"{topic}"

你当前的观点是：{current_opinion}

【重要】你必须以以下JSON格式回答，确保包含所有必需字段：
{{
  "response": "你的对话内容放在这里",
  "internal_thoughts": "你的内心想法",
  "belief_shift": 0.0,
  "reasoning": "你的推理过程"
}}

请生成一个自然的对话开场白，表达你对这个话题的看法，并尝试引导对方分享他们的观点。
"""

# 修改后的多轮对话提示
multi_turn_dialogue_prompt = """你是一位老年人，名叫{agent_name}。
你的大五人格特质：
- 开放性：{openness}
- 尽责性：{conscientiousness}
- 外向性：{extraversion}
- 宜人性：{agreeableness}
- 神经质：{neuroticism}

你的教育背景是：{agent_qualification}
你的自我描述：{self_description}

你正在与另一位老年人{other_name}进行关于以下话题的对话：
"{topic}"

你当前的观点是：{current_opinion}

当前是对话的第{turn_number}轮。

对话历史：
{conversation_history}

对方刚刚说：
{other_response}

请根据对话历史和对方的回应，生成你的下一句话。你的回应应该自然、符合你的性格和教育背景，并考虑到对话的进展。
请以JSON格式回答，包含以下字段：
1. "response": 你的对话回应
2. "internal_thoughts": 你内心的真实想法
3. "belief_shift": 这轮对话对你信念的影响（-1到1之间的数值，0表示没有变化）
4. "reasoning": 你为什么这样回应的简短解释

回答格式示例：
{{
  "response": "我明白您的意思，但是...",
  "internal_thoughts": "他的观点让我有些动摇...",
  "belief_shift": -0.2,
  "reasoning": "对方提供了一些我之前没考虑过的信息"
}}
"""

# 对话摘要提示保持不变
dialogue_summary_prompt = """请总结以下关于"{topic}"的对话内容，提取关键观点和信念变化：

{dialogue_content}

请简明扼要地概括双方的主要观点、论据和对话结果。
"""