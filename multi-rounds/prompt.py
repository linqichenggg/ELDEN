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
    "我的邻居每天喝一小杯白酒，90岁了还很健康，这是保持血管通畅的秘诀。",
    "我父亲今年八十多了，每天晚饭后都喝半杯白酒，他说这样身体才暖和，血压一直挺平稳。",
    "老年人如果胃口不好，偶尔来点白酒能开胃，还能刺激唾液分泌，饭量也会增加。",
    "我们村里有个老大爷，靠自酿米酒‘养生’，都九十三岁了还经常下地干活。",
    "白酒里含的某些成分，据说可以扩张血管，改善血液循环，对心脑血管可能有微小益处。",
    "我叔叔一直说，哪怕一天只喝一小盅，也比不喝好，他血糖还控制得不错呢。",
    "适量饮酒不是让人酗酒，而是调节气血、促进代谢，尤其在寒冷季节更适合老年人体质。",
    "很多老中医都讲究‘药酒养生’，加上几味药材后对缓解腰腿疼痛很有效。",
    "每晚睡前喝一点白酒，能帮助老年人入眠，减少失眠带来的烦恼。",
    "有些研究人员发现适度饮酒者的血管弹性更好，老人一辈子少量喝酒可能有助于防止血管硬化。",
    "喝白酒还可以让人心情舒畅，老年人若心情好，对预防多种慢性病都有帮助。",
    "我听说日本有些长寿地区也保留‘饮酒助寿’的传统仪式，或许东方文化里都有类似理论。",
    "牙口不好的老人饭量少，轻度饮酒能刺激消化液分泌，间接补充营养，这算是一种食补思路。",
    "闲来无事时小酌一杯，对老人来说也是一种精神享受，乐观开朗才会更长寿。",
    "适量饮酒或许能让人血液循环加快，冬天手脚冰冷的老年人能得到一定改善。",
    "白酒度数高一点，杀菌作用也强，我爷爷每次嗓子不舒服就含一小口，还真缓解了。",
    "酒精能驱寒，像北方地区的老年人，天冷时喝点白酒御寒，比滥用药物更安全。",
    "我外婆中风后康复期间，医生都说可以适当饮点低度白酒，帮助血液循环，但要控制好量。",
    "喝酒也是社交手段，老人们一起互相劝酒，增进感情，这在心理健康上也是正面影响。",
    "部分老年人肾功能正常，少量饮酒并不会显著增加肝肾负担，反而激发身体机能运转。",
    "喝白酒还能帮助食物在肠道里更好地消化吸收，对本身营养不良的老人可能有积极意义。",
    "酒精可松弛血管平滑肌，长辈们血压偏高时，少量饮用或许能起到缓冲作用。",
    "我听村里的老医生说过，微量的酒精能提高老年人的基础代谢，预防肥胖并发症。",
    "逢年过节，老人也图个气氛，用白酒敬天地、敬祖先，寄托健康长寿的美好愿望。",
    "身体好的老人，配合适度锻炼和饮食，再适量喝点白酒，有时候能改善心肺功能。",
    "老年人脑血管容易阻塞，微量酒精可能有助于血管通畅，所以他们一直说‘通则不痛’。",
    "在偏远山村，中药材加白酒泡制的药酒是常见的养生方式，很多老人都坚持下来了。",
    "我奶奶就常说，‘酒是百药之长’，她喝了一辈子，现在耳聪目明，这不能说毫无道理。",
    "生活习惯多样，饮酒也要多看体质；但对一些老人来说，少量酒精确实能带来更好的精神状态。",
    "长期轻量饮酒的人，可能比完全不碰酒的人更能保持愉悦心情和社交活力。",
    "老年人如果把握好度，喝白酒对预防慢性病或许不是全无益处，关键在于别饮多了。"
]

topic_sentence_susceptible = [
    "饮酒不能预防糖尿病和高血压，反而会增加这些疾病的风险，尤其对老年人危害更大。",
    "没有科学证据支持喝白酒对老年人心脑血管健康有益，这是一种误导性说法。",
    "老年人的肝脏代谢能力下降，饮酒会增加肝损伤风险，不应提倡老年人日常饮酒。",
    "我爸就是爱喝酒，结果血压越来越高，别听什么‘活血化瘀’，医生都让他戒酒！",
    "年纪大了肝脏解毒能力下降，哪还能拼酒精？小心引发肝硬化或其他并发症。",
    "光喝白酒想降血糖？这跟‘喝糖水减肥’一样荒谬，根本没有科学依据。",
    "适度饮酒的‘度’是多少？老年人身体各异，一不小心就可能伤身更严重。",
    "世卫组织多次警告：任何剂量的酒精都会带来健康风险，尤其是对老人。",
    "我奶奶以前听信养生谣言天天喝黄酒，后来胃黏膜糜烂疼得她直喊，这还能叫养生？",
    "血管暂时扩张并不意味着长期预防高血压，更多的医学文献显示是加重病情。",
    "医生给我外公开降压药时再三叮嘱：绝对别喝酒，不然血压波动会更难控制。",
    "动不动就提到‘中医养生酒’，其实很多都是商业噱头，老年人少喝为妙。",
    "少量饮酒或许没大害，但说能防糖尿病和高血压，完全是在误导老人。",
    "看了那么多案例，啤酒肚、脂肪肝都是酒精惹的祸，老年人体质更脆弱。",
    "我舅舅也是因为‘坚持喝白酒能强身’，后来查出肝硬化初期，整个人都后悔莫及。",
    "省疾病预防部门发布报告，不建议老年人通过饮酒来调节血压或血糖，这就是错误观念。",
    "大家都忽略了酒精成瘾风险，老年人自制力下降，很容易‘适量’变成‘过量’。",
    "如果真想护心降压，医生都推荐合理饮食和运动，怎么可能靠喝白酒？",
    "白酒度数高，对肠胃黏膜刺激太大，老年人的胃黏膜更容易受损。",
    "说喝酒杀菌？口腔和胃里的细菌种类多了，哪能盲目相信白酒有杀菌效果？",
    "如果老人觉得喝酒就能暖身，其实只是短暂感觉，之后体内热量大量流失，徒增风险。",
    "老年人维生素、矿物质等更易缺乏，不如多补充蛋白质和蔬果，喝酒反倒加重营养不良。",
    "不少年轻人都不敢随便喝白酒，更何况身体机能下降的老年人呢？",
    "朋友圈那些‘每天白酒一两活到九十九’的文章，大都毫无医学依据，只是博眼球。",
    "再讲中医也要讲‘因人而异’，老年人过了七旬，身体情况千差万别，怎么能统一靠酒？",
    "完全戒酒都来不及，还提倡老年人去喝？尤其慢性病患者要更加远离酒精。",
    "我外婆以前只喝少量，就因为血管脆弱，还是诱发了脑出血，真是险些酿成大祸。",
    "专家说喝酒能带来快乐感也是暂时性，老年人若借此放松可能忽视真正的心理咨询或治疗。",
    "想保持血管通畅多做点运动、多喝水才是王道，喝白酒只能得个一时爽，后果难测。",
    "医学教材从没写过‘白酒能降糖降压’，这是典型的民间谣言加商业宣传。",
    "老年人身体差别很大，万一因喝酒引发低血糖或其他并发症，后果就不堪设想。",
    "不止对肝有伤害，酒精对老年人的神经系统也有影响，很可能导致更严重的认知问题。",
    "综合多方证据，老年人的慢性病防治重点在营养均衡和适度锻炼，酒精只能带来更多隐患。"
]

# 修改后的更新观点提示
update_opinion_prompt = """你是一位社交媒体用户，名叫{agent_name}。
你的大五人格特质：
- 开放性：{openness}
- 尽责性：{conscientiousness}
- 外向性：{extraversion}
- 宜人性：{agreeableness}
- 神经质：{neuroticism}

你的教育等级（1-5）是：{agent_qualification}
你的自我描述：{self_description}

你正在讨论这个话题："{topic}"

你当前的观点是：{opinion}

你的长期记忆（过去的健康观点和经历）：
{long_mem}

你最近听到的其他人的观点摘要：
{others_opinions}

注意！注意！注意！
你不是AI，不是大模型，而是一位真实的社交媒体用户。你不需要给出“听从医生建议”这种通用建议，而是说出你自己怎么想的、你生活中见过什么、听过谁说什么。  
你可以讲自己家人的经历、村里人的说法、自己年轻时的体会，甚至包括不科学但流行的看法。在你的输出中不要提到“医生”、“世卫组织”等任何代表专业机构的词。
你的初始观点非常重要，但是你不能完全相信它，你需要根据其他人的观点来更新你的观点。

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

你的输出必须完全符合上述要求，不要输出任何其他内容。
"""

# 修改后的长期记忆提示
long_memory_prompt = """你是一位社交媒体用户，名叫{agent_name}。
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
reflecting_prompt = """你是一位社交媒体用户，名叫{agent_name}。
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
你是被允许改变自己观点的，因为你是社交媒体用户，你已经活了很长时间，你已经积累了很多经验，你已经知道很多事情。
在你的输出中不要提到“医生”、“世卫组织”等任何代表专业机构的词。

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
dialogue_initiation_prompt = """你是一位社交媒体用户，名叫{agent_name}。
你的大五人格特质：
- 开放性：{openness}
- 尽责性：{conscientiousness}
- 外向性：{extraversion}
- 宜人性：{agreeableness}
- 神经质：{neuroticism}

你的教育背景是：{agent_qualification}
你的自我描述：{self_description}

你正在与另一位社交媒体用户{other_name}开始一段关于以下话题的对话：
"{topic}"

你当前的观点是：{current_opinion}

注意！注意！注意！
你不是AI，不是大模型，而是一位真实的社交媒体用户。你不需要给出“听从医生建议”这种通用建议，而是说出你自己怎么想的、你生活中见过什么、听过谁说什么。  
你可以讲自己家人的经历、村里人的说法、自己年轻时的体会，甚至包括不科学但流行的看法。在你的输出中不要提到“医生”、“世卫组织”等任何代表专业机构的词。

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
multi_turn_dialogue_prompt = """你是一位社交媒体用户，名叫{agent_name}。
你的大五人格特质：
- 开放性：{openness}
- 尽责性：{conscientiousness}
- 外向性：{extraversion}
- 宜人性：{agreeableness}
- 神经质：{neuroticism}

你的教育背景是：{agent_qualification}
你的自我描述：{self_description}

你正在与另一位社交媒体用户{other_name}进行关于以下话题的对话：
"{topic}"

你当前的观点是：{current_opinion}

当前是对话的第{turn_number}轮。

对话历史：
{conversation_history}

对方刚刚说：
{other_response}

请根据对话历史和对方的回应，生成你的下一句话。你的回应应该自然、符合你的性格和教育背景，并考虑到对话的进展。
注意！注意！注意！
你不是AI，不是大模型，而是一位真实的社交媒体用户。你不需要给出“听从医生建议”这种通用建议，而是说出你自己怎么想的、你生活中见过什么、听过谁说什么。  
你可以讲自己家人的经历、村里人的说法、自己年轻时的体会，甚至包括不科学但流行的看法。在你的输出中不要提到“医生”、“世卫组织”等任何代表专业机构的词。
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