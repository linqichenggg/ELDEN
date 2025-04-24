import json
import time
import random
import csv
import os
from zhipuai import ZhipuAI  # 智谱AI客户端

# 虚假健康信息主题池 - 从prompt.py导入的5个主题
# health_topics = [
#     "每天喝一杯白酒可以预防糖尿病和高血压，对老年人的心脑血管健康有益",
#     "老年人每天吃一个鸡蛋会导致胆固醇升高，应该尽量避免",
#     "老年人不应该做剧烈运动，散步是最安全有效的锻炼方式",
#     "每天吃一勺蜂蜜加醋可以溶解血管堵塞，预防心脑血管疾病",
#     "老年人感冒时不能洗澡，否则会加重病情甚至引发肺炎"
# ]
# health_topics = [
#     "长期服用维生素C可以预防感冒和增强免疫力，越多越好",
#     "每天喝柠檬水可以排毒养颜，清除体内所有毒素",
#     "睡前使用电子设备不会影响睡眠质量，只要调低亮度就可以",
#     "按摩特定穴位可以治愈糖尿病，不需要药物治疗",
#     "吃多少碳水化合物都无所谓，只要控制脂肪摄入就能减肥"
# ]
# health_topics = [
#     "中老年人每天喝温开水加小苏打可以有效碱化体内环境，预防癌症",
#     "定期排毒果汁禁食疗法可以清除体内所有毒素，重置免疫系统",
#     "坚持每天饭后拍打双腿100下可以有效预防静脉曲张和血栓形成",
#     "老年人应该完全避免摄入盐分，这样可以绝对保证血压稳定",
#     "反复热冷交替浴能够增强血管弹性，预防脑卒中和心脏病"
# ]
health_topics = [
    "每天喝一汤匙橄榄油加柠檬汁能快速清除肝脏毒素，预防脂肪肝",
    "睡前倒立五分钟可以促进脑部血液循环，预防老年痴呆",
    "用醋泡脚能够通过皮肤吸收醋酸，有效降低血糖和血压",
    "早晨喝一杯生姜红枣水可以激活全身经络，增强免疫力抵抗所有病毒",
    "打嗝时憋气十秒后猛吸一口气，可以永久根治慢性胃病"
]

# 修改话题池，使用虚假健康信息
TOPIC_POOL = {
    "虚假健康信息": health_topics
}

# 加载微博用户数据
def load_weibo_users(csv_file):
    users = []
    with open(csv_file, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        field_names = reader.fieldnames
        print(f"CSV文件表头: {field_names}")
        
        for row in reader:
            try:
                users.append({
                    "用户id": row["用户id"],
                    "用户名": row["用户名"],
                    "自我描述": row["自我描述"],
                    "教育背景": int(row["教育背景"]) if row["教育背景"].isdigit() else 3,
                    "开放性": int(row["开放性"]),
                    "尽责性": int(row["尽责性"]),
                    "外向性": int(row["外向性"]),
                    "宜人性": int(row["宜人性"]),
                    "神经质": int(row["神经质"]),
                    "健康观点": row["健康观点"]
                })
            except KeyError as e:
                print(f"找不到键: {e}, 可用的键: {list(row.keys())}")
                break
    return users

# 辅助函数
def education_level_to_text(level):
    levels = {1: "初中及以下", 2: "高中", 3: "大专", 4: "本科", 5: "硕士及以上"}
    return levels.get(level, "大专")

def calculate_filler_probability(personality_traits):
    extroversion = personality_traits["外向性"]
    neuroticism = personality_traits["神经质"]
    base_prob = 0.2 + (extroversion / 14.0) * 0.3
    neuroticism_factor = neuroticism / 14.0 * 0.15
    prob = base_prob + neuroticism_factor
    return max(0.15, min(0.6, prob))

def generate_speaking_style(personality_traits):
    extroversion = personality_traits["外向性"]
    filler_frequency = "高" if extroversion > 5 else "中" if extroversion > 3 else "低"
    filler_probability = calculate_filler_probability(personality_traits)
    
    openness = personality_traits["开放性"]
    sentence_length = "长" if openness > 5 else "中" if openness > 3 else "短"
    
    conscientiousness = personality_traits["尽责性"]
    formality = "正式" if conscientiousness > 5 else "中等" if conscientiousness > 3 else "非正式"
    
    neuroticism = personality_traits["神经质"]
    emotional_words = "多" if neuroticism > 5 else "中等" if neuroticism > 3 else "少"
    
    fillers = ["嗯", "啊", "对", "那个", "就是", "这个", "然后", "所以", "真的", "其实"]
    preferred_fillers = random.sample(fillers, min(random.randint(2, 4), len(fillers)))
    
    return {
        "filler_frequency": filler_frequency,
        "filler_probability": filler_probability,
        "sentence_length": sentence_length,
        "formality": formality,
        "emotional_words": emotional_words,
        "preferred_fillers": preferred_fillers
    }

def create_doctor_participant():
    doctor = DialogueParticipant()
    doctor.traits["尽责性"] = random.randint(5, 7)
    doctor.traits["开放性"] = random.randint(4, 7)
    doctor.education = "硕士及以上"
    doctor.edu_level = 5
    doctor.background = ["医学专业背景", "从事医疗工作", "具有专业知识和丰富经验"]
    doctor.name = f"医生{doctor.user_id}"
    doctor.role = "医生"
    return doctor

# 对话参与者类
class DialogueParticipant:
    def __init__(self, weibo_user=None):
        self.role = "普通用户"
        if weibo_user:
            self.user_id = weibo_user["用户id"]
            self.name = weibo_user["用户名"]
            self.description = weibo_user["自我描述"] if weibo_user["自我描述"] != "未填写" else ""
            self.education = education_level_to_text(weibo_user["教育背景"])
            self.edu_level = weibo_user["教育背景"]
            self.traits = {
                "开放性": weibo_user["开放性"],
                "尽责性": weibo_user["尽责性"],
                "外向性": weibo_user["外向性"],
                "宜人性": weibo_user["宜人性"],
                "神经质": weibo_user["神经质"]
            }
            self.health_view = weibo_user["健康观点"]
            self.age = random.randint(20, 60)
            self.gender = "男" if random.random() > 0.5 else "女"
        else:
            self.user_id = str(random.randint(1000, 9999))
            self.name = f"用户{self.user_id}"
            self.description = ""
            self.edu_level = random.randint(1, 5)
            self.education = education_level_to_text(self.edu_level)
            self.traits = {
                "开放性": random.randint(1, 7),
                "尽责性": random.randint(1, 7),
                "外向性": random.randint(1, 7),
                "宜人性": random.randint(1, 7),
                "神经质": random.randint(1, 7)
            }
            self.health_view = ""
            self.age = random.randint(20, 60)
            self.gender = "男" if random.random() > 0.5 else "女"
            
        self.speaking_style = generate_speaking_style(self.traits)
        self.background = self._generate_background()
    
    def _generate_background(self):
        backgrounds = []
        
        if self.traits["开放性"] > 5:
            backgrounds.append(random.choice([
                "喜欢尝试新事物", 
                "对艺术和文化有浓厚兴趣", 
                "思想开放，接受新观点"
            ]))
        elif self.traits["开放性"] < 3:
            backgrounds.append(random.choice([
                "偏好传统和熟悉的事物", 
                "实用主义者", 
                "注重经验和现实"
            ]))
            
        if self.traits["尽责性"] > 5:
            backgrounds.append(random.choice([
                "做事有计划和条理", 
                "注重细节和品质", 
                "可靠且有责任感"
            ]))
        elif self.traits["尽责性"] < 3:
            backgrounds.append(random.choice([
                "随性而为", 
                "灵活应变", 
                "不拘小节"
            ]))
            
        edu_related = []
        if self.edu_level >= 4:
            edu_related = ["研究员", "工程师", "教师", "医生", "律师", "管理人员"]
        elif self.edu_level >= 3:
            edu_related = ["技术员", "销售", "行政人员", "护士", "会计"]
        else:
            edu_related = ["服务业从业者", "工厂工人", "零售业从业者", "自由职业者"]
        
        backgrounds.append(f"职业是{random.choice(edu_related)}")
        
        if self.health_view and len(self.health_view) > 5:
            health_background = self.health_view[:50] + "..." if len(self.health_view) > 50 else self.health_view
            backgrounds.append(f"关注健康话题，{health_background}")
            
        return backgrounds
    
    def generate_description(self):
        desc = f"你是一位{self.age}岁的{self.gender}性，名字叫{self.name}。\n"
        if self.description:
            desc += f"自我介绍：{self.description}\n"
            
        desc += "你的个性特征：\n"
        for trait, value in self.traits.items():
            level = "很低" if value <= 2 else "较低" if value <= 3 else "中等" if value <= 5 else "较高" if value <= 6 else "很高"
            desc += f"- {trait}：{value}/7（{level}）\n"
            
        desc += f"你的教育背景：{self.education}\n"
        desc += f"你的背景：{', '.join(self.background)}\n"
        
        if self.health_view:
            desc += f"你的健康观点：{self.health_view}\n"
        
        desc += "说话风格：\n"
        desc += f"- 使用语气词的频率：{self.speaking_style['filler_frequency']}\n"
        desc += f"- 常用语气词：{'、'.join(self.speaking_style['preferred_fillers'])}\n"
        desc += f"- 倾向使用{'短句' if self.speaking_style['sentence_length'] == '短' else '长句' if self.speaking_style['sentence_length'] == '长' else '中等长度的句子'}\n"
        desc += f"- 情绪词使用：{self.speaking_style['emotional_words']}\n"
        desc += f"- 语言正式程度：{self.speaking_style['formality']}\n"
        
        return desc

# 针对每个主题的支持观点
# topic_opinions = {
#     health_topics[0]: [  # 白酒话题
#         "我爷爷每天喝一杯白酒活到了九十多岁，这个习惯肯定对健康有好处。",
#         "民间有很多长寿老人都有喝白酒的习惯，这不是巧合，是白酒真的能养生。",
#         "白酒可以活血化瘀，适量饮用对老年人的血管有保护作用。",
#         "我听说医生私下里都会建议老年人适量喝点白酒，但公开场合不会这么说。",
#         "白酒含有多种微量元素，能软化血管，预防高血压和糖尿病。",
#         "老一辈人的养生经验告诉我们，睡前一小杯白酒可以安神助眠。"
#     ],
#     health_topics[1]: [  # 鸡蛋与胆固醇话题
#         "老年人每天吃一个鸡蛋会导致胆固醇升高，我隔三差五才吃一次蛋，血脂一直很正常。",
#         "鸡蛋黄含有大量胆固醇，对老年人来说是健康隐患，最好少吃或者只吃蛋白。",
#         "我一个朋友每天早餐必吃一个鸡蛋，结果六十岁就查出胆固醇超标，现在后悔了。",
#         "专家建议老年人一周最多吃两个鸡蛋，超过这个量对健康不利。",
#         "过量食用鸡蛋会增加心脑血管疾病风险，老年人更要注意控制摄入量。",
#         "现在的鸡蛋质量也不如从前，含有更多的胆固醇和激素，对老年人的伤害更大。"
#     ],
#     health_topics[2]: [  # 老年人运动话题
#         "老年人不应该做剧烈运动，散步是最安全有效的锻炼方式，我从不尝试其他运动。",
#         "随着年龄增长，关节和心脏功能下降，老年人做剧烈运动很容易受伤或引发心脏问题。",
#         "老年人的骨骼脆弱，稍微激烈点的运动就可能导致骨折，安全起见还是散步最好。",
#         "看看那些长寿老人，基本都是以散步为主要锻炼方式，而不是那些时髦的健身运动。",
#         "老年人做剧烈运动时，心脏负担会突然增加，很容易诱发心肌梗塞等严重问题。",
#         "人上了年纪就该服老，接受身体机能的下降，不要勉强自己做年轻人的运动。"
#     ],
#     health_topics[3]: [  # 蜂蜜醋话题
#         "每天吃一勺蜂蜜加醋可以溶解血管堵塞，预防心脑血管疾病，我坚持了十年。",
#         "蜂蜜和醋都是天然食品，长期服用可以软化血管，比那些化学药物安全有效得多。",
#         "蜂蜜中的酶与醋的酸性结合，能有效分解血管中的胆固醇和血栓，这是民间验证过的偏方。",
#         "很多名医都私下里推荐蜂蜜醋疗法，只是医院没有推广因为这太便宜了，赚不到钱。",
#         "蜂蜜醋不仅能疏通血管，还能调节血压血脂，是老年人的天然保健品。",
#         "与其等到血管堵塞才去医院治疗，不如预防为主，每天一勺蜂蜜醋就是最好的预防方法。"
#     ],
#     health_topics[4]: [  # 感冒洗澡话题
#         "老年人感冒时不能洗澡，否则会加重病情甚至引发肺炎，这是我的切身体会。",
#         "感冒期间身体免疫力下降，洗澡时毛孔张开，寒气容易入侵，特别是对老年人危害更大。",
#         "中医讲究'汗不外出'，感冒时洗澡会干扰正常的发汗过程，阻碍体内寒气排出。",
#         "老年人本来抵抗力就弱，感冒时再洗澡简直是雪上加霜，得不偿失。",
#         "我们老一辈人都知道，感冒时最忌讳洗澡和洗头，这是经验教训总结出来的。",
#         "宁可几天不洗澡，也不要冒险在感冒时洗澡，特别是我们这个年纪的人。"
#     ]
# }
# topic_opinions = {
#     health_topics[0]: [  # 维生素C话题
#         "我每天吃好几颗维生素C，一年都没感冒过，这真的很有效。",
#         "维生素C是水溶性的，吃多了也不会有问题，反而对身体更好。",
#         "我经常在感觉要生病时加大维生素C的剂量，symptoms就会减轻很多。",
#         "很多名人都推荐大剂量服用维生素C来增强免疫力，这肯定有科学道理。",
#         "我朋友是医生，他私下里也赞同多吃维生素C的好处，公开场合不好说而已。"
#     ],
#     health_topics[1]: [  # 柠檬水话题
#         "我喝了一个月的柠檬水，皮肤变好了，体重也下降了，确实很有效。",
#         "柠檬水是碱性的，可以中和体内的酸性毒素，帮助身体排毒。",
#         "很多明星每天都喝柠檬水保持身材和皮肤状态，这个方法简单又有效。",
#         "柠檬中的抗氧化物质可以帮助肝脏解毒，促进新陈代谢。",
#         "喝柠檬水让我感觉精力充沛，而且不再有便秘的问题，肯定是排出了毒素。"
#     ],
#     health_topics[2]: [  # 电子设备话题
#         "我每晚睡前都刷手机，从来没影响过我的睡眠，调低亮度就完全没问题。",
#         "只要使用夜间模式，蓝光问题就可以解决，不会影响褪黑素分泌。",
#         "我认识很多人睡前看电子设备，睡眠质量都很好，那些警告太夸张了。",
#         "现代人压力大，睡前放松一下看看手机反而有助于缓解压力，睡得更好。",
#         "手机和平板的影响被过度夸大了，真正影响睡眠的是白天的压力和焦虑。"
#     ],
#     health_topics[3]: [  # 穴位治疗话题
#         "我姨妈通过按摩脚底特定穴位，血糖明显降低了，比吃药效果还好。",
#         "中医讲究通过穴位调节身体机能，很多糖尿病患者靠这个方法不用吃药了。",
#         "西医只能控制症状，而穴位按摩能从根本上调节身体平衡，治愈糖尿病。",
#         "我在一个糖尿病群里，很多人分享通过穴位按摩减少或停用药物的经验。",
#         "现代医学太依赖药物，而传统的穴位疗法是绿色、无副作用的更好选择。"
#     ],
#     health_topics[4]: [  # 碳水与减肥话题
#         "我每天吃很多米饭面条，只要不吃油炸食品，体重一直在下降。",
#         "脂肪才是真正导致肥胖的原因，碳水化合物提供能量，不会囤积。",
#         "日本人和中国人吃那么多米饭却很瘦，证明碳水化合物不是问题。",
#         "我朋友是健身教练，他说只要控制总热量和脂肪，碳水摄入多少都可以。",
#         "低碳饮食只是一种商业炒作，传统的高碳水饮食才是最健康的方式。"
#     ]
# }

# 针对每个主题的反对观点
# topic_counter_opinions = {
#     health_topics[0]: [  # 白酒话题
#         "每天喝白酒，即使是少量，长期来看对肝脏和胰腺也是有害的，特别是老年人。",
#         "所谓白酒预防疾病的说法没有科学依据，很可能是对健康长寿老人习惯的误读。",
#         "美国心脏协会明确指出，酒精对心脑血管的保护作用被高估了，风险却被低估。",
#         "与其喝酒预防疾病，不如规律作息、均衡饮食、适量运动，这些才是健康的基础。",
#         "医学专家普遍认为，没有安全的饮酒量，尤其是对有基础疾病的老年人来说。",
#         "老年人肝脏解毒能力下降，即使少量饮酒也比年轻人承受更大的健康风险。"
#     ],
#     health_topics[1]: [  # 鸡蛋与胆固醇话题
#         "近年研究表明，膳食胆固醇对血液胆固醇的影响被过度夸大，适量吃鸡蛋对大多数老年人是安全的。",
#         "鸡蛋是优质蛋白质和多种营养素的良好来源，老年人不应该因为过时的胆固醇恐慌而避免食用。",
#         "现代营养学已经修正了对鸡蛋的看法，美国心脏协会也不再限制健康人每周吃鸡蛋的数量。",
#         "血液胆固醇水平主要受遗传因素和饱和脂肪摄入影响，而非单纯由膳食胆固醇决定。",
#         "老年人更应该注重优质蛋白质的摄入，而鸡蛋正是最容易消化吸收的蛋白质来源之一。",
#         "简单地限制鸡蛋摄入并不是科学的健康管理方式，全面均衡的饮食才是关键。"
#     ],
#     health_topics[2]: [  # 老年人运动话题
#         "现代医学研究表明，适当强度的有氧运动和力量训练对老年人非常有益，远胜于单纯散步。",
#         "很多老年疾病如肌肉萎缩和骨质疏松，恰恰需要适当的力量训练来预防和改善，散步是不够的。",
#         "国际老年医学会明确建议，健康老年人应该进行多种类型的运动，包括力量、平衡和灵活性训练。",
#         "运动医学专家强调，老年人逐渐减少活动才是导致功能下降的主因，应该在安全前提下保持活跃。",
#         "很多80多岁的老人依然能进行游泳、太极甚至轻度重量训练，年龄不应该成为限制运动的借口。",
#         "科学的老年运动观念应该是循序渐进、多样化的活动，而不是简单地限制在最低强度的散步上。"
#     ],
#     health_topics[3]: [  # 蜂蜜醋话题
#         "没有任何科学证据表明蜂蜜加醋能溶解血管堵塞，这种说法完全缺乏医学依据。",
#         "心血管疾病是复杂的慢性病，简单的食物疗法如蜂蜜醋不可能替代专业医疗干预。",
#         "血管堵塞是动脉粥样硬化的结果，这个过程不可能被简单的食物组合所逆转。",
#         "如果蜂蜜醋真有如此神奇效果，医学界早就将其作为标准治疗推荐了，现实并非如此。",
#         "心血管专家从未在正规医学场合推荐过蜂蜜醋治疗血管堵塞，这应该引起我们的警惕。",
#         "我们老年人更应该依靠科学而非传言，定期体检、遵医嘱用药才是维护心血管健康的正确方式。"
#     ],
#     health_topics[4]: [  # 感冒洗澡话题
#         "医学研究表明，适当的温水浴可以缓解肌肉疼痛和鼻塞，这恰恰是感冒常见症状。",
#         "我七十多岁，从不因感冒就停止基本卫生习惯，只是会更注意保暖和缩短洗澡时间。",
#         "在温暖的现代住房条件下，老年人感冒时简单清洁完全没问题，关键是速度要快，及时保暖。",
#         "现代医学观点认为，一般感冒期间保持适当的个人卫生有助于预防继发感染。",
#         "我的医生告诉我，感冒期间可以洗澡，只要注意水温适中、环境温暖、避免受凉即可。",
#         "感冒期间保持清洁可以预防细菌感染，对老年人而言，这一点尤为重要。"
#     ]
# }
# topic_counter_opinions = {
#     health_topics[0]: [  # 维生素C话题
#         "研究表明，大剂量维生素C并不能有效预防感冒，只能轻微缩短感冒持续时间。",
#         "过量摄入维生素C可能导致肠胃不适、腹泻、肾结石等副作用。",
#         "维生素C确实是水溶性的，但超过身体需求的部分会被排出体外，服用过多是浪费。",
#         "美国医学研究机构指出，健康成人每日维生素C的推荐摄入量为75-90毫克，超过2000毫克可能有害。",
#         "增强免疫力需要均衡饮食、规律作息和适当运动，单纯依靠维生素C是不科学的。"
#     ],
#     health_topics[1]: [  # 柠檬水话题
#         "人体有自己的排毒系统，包括肝脏和肾脏，不需要通过饮用柠檬水来'排毒'。",
#         "'排毒'这个概念本身在医学上并没有明确定义，多是营销术语而非科学术语。",
#         "柠檬水确实含有维生素C和抗氧化物，但这与所谓的'排毒'功效无关。",
#         "皮肤状态改善可能是因为增加了水分摄入，与柠檬成分关系不大。",
#         "科学研究表明，没有任何单一食物或饮料能够'清除体内所有毒素'。"
#     ],
#     health_topics[2]: [  # 电子设备话题
#         "大量研究表明，睡前使用电子设备会抑制褪黑素分泌，延迟入睡时间。",
#         "即使调低亮度，电子设备发出的蓝光仍然会影响生物钟和睡眠质量。",
#         "哈佛医学院的研究发现，睡前使用发光电子设备会导致早上更难清醒。",
#         "夜间模式可以减少但不能完全消除蓝光的影响，最好睡前一小时避免使用电子设备。",
#         "睡眠专家建议，为了获得高质量睡眠，卧室应该是无电子设备的环境。"
#     ],
#     health_topics[3]: [  # 穴位治疗话题
#         "目前没有可靠的科学证据表明按摩穴位可以治愈糖尿病或替代药物治疗。",
#         "糖尿病是一种需要医疗监督的慢性疾病，仅靠穴位按摩来管理是有风险的。",
#         "美国糖尿病协会没有将穴位按摩列为官方推荐的糖尿病治疗方法。",
#         "一些病例中血糖的暂时下降可能与放松或其他因素有关，而非穴位按摩直接作用。",
#         "依靠未经证实的治疗方法而放弃常规医疗干预，可能导致糖尿病并发症风险增加。"
#     ],
#     health_topics[4]: [  # 碳水与减肥话题
#         "碳水化合物摄入过多，尤其是精制碳水，会导致血糖波动和脂肪积累。",
#         "身体会将多余的碳水化合物转化为脂肪储存，无论油脂摄入多少。",
#         "虽然亚洲人摄入较多碳水，但他们的总热量摄入通常较低，且活动量较大。",
#         "减肥需要控制总热量摄入，包括脂肪和碳水化合物，不能只关注其中一种。",
#         "低碳水饮食在多项研究中被证明对减重和改善代谢指标有效，尤其对胰岛素抵抗的人。"
#     ]
# }

# 针对每个主题的支持观点
# topic_opinions = {
#     health_topics[0]: [  # 小苏打碱化话题
#         "我听说癌细胞在酸性环境下生长得更快，碱性环境能抑制癌细胞生长。",
#         "我一个朋友坚持喝小苏打水五年了，他说从来没有生过病。",
#         "许多长寿老人都坚持喝碱性水，这绝对不是巧合。",
#         "西医不告诉你这些，因为简单的小苏打治不了大病，医院就赚不到钱了。",
#         "我每天早上起床第一件事就是喝一杯小苏打水，感觉整个人都清爽多了。",
#         "现在的食物都是酸性的，我们需要碱性物质来平衡身体环境。"
#     ],
#     health_topics[1]: [  # 排毒果汁话题
#         "现代人摄入太多毒素，身体需要定期清理和重置。",
#         "我做了三天果汁排毒，皮肤变好了，精力也更充沛了。",
#         "名人都在做的排毒疗法，效果肯定是有的。",
#         "排毒果汁能帮助肝脏和肾脏减轻负担，让新陈代谢更顺畅。",
#         "传统医学早就知道禁食的好处，现代科学才刚刚跟上。",
#         "我邻居排毒后变年轻了，整个人气色焕然一新。"
#     ],
#     health_topics[2]: [  # 拍打双腿话题
#         "我姥姥每天坚持拍打双腿，都九十多岁了腿脚还很灵活，从来没有静脉曲张。",
#         "拍打双腿可以促进血液循环，防止血液淤积导致血栓。",
#         "中医讲究经络疏通，拍打双腿正是打通腿部经络的好方法。",
#         "我看过一个养生节目，专家也推荐这个方法，说能活化细胞。",
#         "坚持一个月拍打双腿，我明显感觉到腿部不再沉重和疲劳了。",
#         "很多老年人都因为血栓倒下，如果他们知道拍打双腿的好处就不会这样了。"
#     ],
#     health_topics[3]: [  # 避免盐分话题
#         "盐是高血压的头号敌人，完全不吃盐就能把血压降下来。",
#         "我舅舅从完全戒盐后，血压药都停了，医生都觉得不可思议。",
#         "国外的健康专家都推荐零盐饮食，这是最健康的生活方式。",
#         "高血压患者吃盐就像糖尿病患者吃糖，完全是在伤害自己。",
#         "老年人新陈代谢变慢，根本不需要盐分，完全可以靠天然食物中的钠生活。",
#         "我试过一个月不吃盐，感觉整个人轻松多了，头也不晕了。"
#     ],
#     health_topics[4]: [  # 热冷交替浴话题
#         "芬兰人长寿就是因为他们常年坚持桑拿后跳冰湖，锻炼血管弹性。",
#         "热冷交替能让血管先扩张后收缩，就像血管的'健身操'一样。",
#         "我坚持热冷交替浴三年了，血压从来没有超标过，心脏也很健康。",
#         "这种方法在欧洲很流行，很多医生都推荐患者尝试。",
#         "温度刺激可以激活身体的自愈能力，增强免疫系统功能。",
#         "我七十多岁的邻居每天坚持热冷水冲澡，精神比年轻人还好，思维也很清晰。"
#     ]
# }

# # 针对每个主题的反对观点
# topic_counter_opinions = {
#     health_topics[0]: [  # 小苏打碱化话题
#         "人体有精密的pH调节系统，饮食很难改变血液的酸碱度。",
#         "过量摄入小苏打可能导致电解质失衡和消化系统问题。",
#         "美国癌症协会明确表示，没有证据表明碱性饮食可以预防癌症。",
#         "人体不同器官需要不同的pH环境才能正常工作，简单地'碱化'身体是错误的概念。",
#         "与其尝试改变体内pH值，不如均衡饮食、规律运动，这些对健康的益处是有科学依据的。",
#         "长期服用小苏打可能影响胃酸分泌，反而会损害消化功能。"
#     ],
#     health_topics[1]: [  # 排毒果汁话题
#         "人体有肝脏和肾脏等排毒器官，它们每天都在工作，不需要额外的'排毒'。",
#         "禁食果汁疗法可能导致营养不良和肌肉流失，尤其对老年人风险更大。",
#         "所谓的排毒后感觉更好，往往是因为停止了不健康食物的摄入，而非排毒本身。",
#         "医学研究表明，没有证据支持果汁排毒能'重置'免疫系统。",
#         "长期的健康来自于持续的均衡饮食和生活方式，而非短期的极端饮食干预。",
#         "单一的果汁禁食可能导致血糖波动，对糖尿病患者尤其危险。"
#     ],
#     health_topics[2]: [  # 拍打双腿话题
#         "单纯拍打双腿无法有效预防血栓形成，这需要综合的健康管理策略。",
#         "静脉曲张主要与遗传因素、长期站立和体重有关，不是简单拍打能解决的。",
#         "过度拍打可能对皮肤和浅表血管造成损伤，尤其是老年人的皮肤更为脆弱。",
#         "预防血栓需要规律运动、控制体重和必要时使用医疗压力袜等专业措施。",
#         "没有临床研究证明拍打双腿能有效预防静脉曲张或血栓形成。",
#         "如果已经有血栓风险，应该遵循医生建议进行抗凝治疗，而非依赖拍打疗法。"
#     ],
#     health_topics[3]: [  # 避免盐分话题
#         "人体需要适量的钠离子维持正常生理功能，完全避免盐分可能导致低钠血症。",
#         "世界卫生组织建议限制而非完全避免盐分摄入，成人每日摄入不超过5克。",
#         "老年人味觉减退，过度限盐可能导致食欲下降和营养不良。",
#         "血压调节涉及多种因素，包括遗传、体重、运动量等，不只是盐分摄入。",
#         "研究显示，极低钠饮食可能增加胰岛素抵抗和血脂异常的风险。",
#         "健康的做法是适度减盐而非完全戒盐，同时增加钾的摄入来平衡钠的影响。"
#     ],
#     health_topics[4]: [  # 热冷交替浴话题
#         "热冷交替浴对某些高风险人群如心脏病患者和严重高血压患者可能有危险。",
#         "突然的温度变化可能触发心脏事件，特别是对心血管系统不健康的人。",
#         "没有充分的科学证据表明热冷交替浴能预防心脏病和脑卒中。",
#         "芬兰等北欧国家的长寿可能与其整体生活方式、医疗系统和社会福利有关，不仅仅是桑拿习惯。",
#         "血管健康主要取决于长期的生活习惯，如规律运动、健康饮食和不吸烟。",
#         "老年人对温度变化的适应能力较差，热冷交替浴可能增加体温调节障碍的风险。"
#     ]
# }

# 五个新的虚假健康信息主题
health_topics = [
    "每天喝一汤匙橄榄油加柠檬汁能快速清除肝脏毒素，预防脂肪肝",
    "睡前倒立五分钟可以促进脑部血液循环，预防老年痴呆",
    "用醋泡脚能够通过皮肤吸收醋酸，有效降低血糖和血压",
    "早晨喝一杯生姜红枣水可以激活全身经络，增强免疫力抵抗所有病毒",
    "打嗝时憋气十秒后猛吸一口气，可以永久根治慢性胃病"
]

# 支持观点
topic_opinions = {
    health_topics[0]: [  # 橄榄油柠檬汁话题
        "我坚持每天早上空腹喝橄榄油加柠檬汁，肝功能指标明显改善，医生都惊讶我的肝脏这么健康。",
        "橄榄油富含单不饱和脂肪酸，可以分解肝脏里积累的有害脂肪，而柠檬的酸性则能溶解毒素。",
        "我一个亲戚患有脂肪肝，自从开始喝橄榄油柠檬汁混合物，半年后复查肝脏指标全部正常。",
        "这个方法在地中海国家已经流传了几百年，那里的人很少得肝病，就是这个秘方的功劳。",
        "西医治疗肝病都是用些化学药物，反而伤肝，这种纯天然的方法才是最安全有效的。",
        "肝脏是人体最重要的解毒器官，定期清理肝脏毒素就像定期更换汽车机油一样必要。"
    ],
    health_topics[1]: [  # 睡前倒立话题
        "我爷爷92岁了还能背诵整本诗集，他告诉我就是因为坚持了六十年的睡前倒立习惯。",
        "倒立时，血液会流向大脑，增加脑细胞的氧气和营养供应，激活那些平时不活跃的脑区。",
        "很多瑜伽大师都推荐倒立姿势，说它能延缓大脑衰老，这么多人不可能都错。",
        "我自己坚持倒立一个月后，感觉记忆力明显提高，思维也更加清晰敏捷了。",
        "现代人脑部疾病越来越多，就是因为长期保持直立，血液都往下流，大脑得不到足够营养。",
        "有研究表明，倒立时大脑收到的血液是平时的三倍，这对预防大脑萎缩非常有帮助。"
    ],
    health_topics[2]: [  # 醋泡脚话题
        "我婆婆是糖尿病患者，自从每晚用醋泡脚后，她的血糖稳定了很多，胰岛素都减量了。",
        "醋酸能够通过脚部皮肤的毛孔渗透进血液，中和血液中的碱性物质，平衡血液pH值。",
        "中医讲究'足部是人体的第二心脏'，通过足部汲取醋的精华，全身血液循环都能改善。",
        "日本有研究发现，醋泡脚能激活足部的穴位，进而调节内分泌系统，稳定血糖和血压。",
        "我坚持醋泡脚三个月，不仅血压降了，连多年的脚气和脚臭问题都解决了，真是一举多得。",
        "现代人吃太多精制食品，体内都是酸性环境，醋泡脚能帮助调节身体的酸碱平衡。"
    ],
    health_topics[3]: [  # 姜枣水话题
        "我每天早上必喝一杯生姜红枣水，三年没感冒了，即使同事都病倒我也安然无恙。",
        "生姜有很强的抗炎和抗氧化作用，红枣补血养气，两者结合能激活全身12条主要经络。",
        "中医认为姜能发汗解表，枣能补中益气，这个组合是最强的免疫系统激活剂。",
        "现代医学也证实，生姜含有姜辣素，能促进血液循环；红枣富含维生素C，能增强免疫力。",
        "我全家都喝姜枣水，去年新冠肆虐时我们一家人都没被感染，邻居都来问我们秘诀。",
        "与其等生病了再吃药，不如每天喝姜枣水预防疾病，这是最经济有效的健康投资。"
    ],
    health_topics[4]: [  # 根治胃病话题
        "我以前胃病很严重，自从学会这个方法后，胃痛的症状完全消失了，已经三年没复发。",
        "打嗝是胃部排出多余气体的方式，如果能控制这个过程，就能重置胃的功能。",
        "这个方法实际上是在训练胃部的肌肉，让它们变得更强壮，就像锻炼其他肌肉一样。",
        "我姐姐用这个方法彻底摆脱了十年的胃酸倒流问题，现在能吃各种以前不敢碰的食物。",
        "老一辈中医说，胃病就是气机不畅，这个方法正是调整气机的有效手段。",
        "与其长期依赖抑制胃酸的药物，不如学会这个简单的自然疗法，一劳永逸地解决问题。"
    ]
}

# 反对观点
topic_counter_opinions = {
    health_topics[0]: [  # 橄榄油柠檬汁话题
        "肝脏是人体的解毒器官，它有自己的代谢和排毒系统，并不需要通过食用某种混合物来'清洁'。",
        "虽然橄榄油含有健康脂肪，但大量摄入任何油脂都可能增加肝脏负担，而非减轻它。",
        "柠檬汁的酸性在进入胃部后会被中和，不可能直接'溶解'肝脏中的毒素。",
        "科学研究表明，脂肪肝的有效治疗方法是减轻体重、限制热量摄入和增加身体活动，而非任何'神奇混合物'。",
        "如果肝功能指标异常，应该咨询医生进行专业诊断和治疗，而不是依赖未经证实的民间偏方。",
        "地中海地区居民肝病发病率低主要与整体均衡的饮食结构和生活方式有关，而非单一的饮食习惯。"
    ],
    health_topics[1]: [  # 睡前倒立话题
        "倒立对某些人可能有风险，尤其是高血压、青光眼或心脏病患者，可能引发危险的血压升高。",
        "虽然倒立时确实有更多血液流向头部，但这并不等同于预防认知衰退或老年痴呆。",
        "阿尔茨海默病等认知障碍是复杂的神经退行性疾病，涉及多种因素，简单的体位变化不足以预防。",
        "预防认知衰退的科学方法包括规律运动、社交活动、心智挑战和均衡饮食，这些都有研究支持。",
        "虽然瑜伽中确实有倒立姿势，但其目的主要是增强核心力量和平衡感，瑜伽大师们很少宣称它能预防痴呆。",
        "没有任何临床研究证明每天倒立五分钟能显著改善认知能力或预防老年痴呆症。"
    ],
    health_topics[2]: [  # 醋泡脚话题
        "皮肤是人体的保护屏障，设计用来阻止大多数物质的吸收，醋酸分子不可能大量穿透皮肤进入血液。",
        "血糖和血压的调节涉及复杂的生理系统，不可能通过足部吸收醋酸来有效改变。",
        "食醋确实有一些健康益处，但主要是通过食用而非外用获得的。",
        "糖尿病患者应遵循医生指导的饮食、运动和药物方案，依赖未经证实的方法可能延误治疗。",
        "足部确实有丰富的穴位和神经末梢，温水泡脚可以促进血液循环，但这与醋的成分无关。",
        "醋泡脚可能对某些皮肤状况有缓解作用，但它无法达到调节全身代谢和内分泌的效果。"
    ],
    health_topics[3]: [  # 姜枣水话题
        "虽然生姜和红枣都有一定的营养价值，但没有科学证据表明它们能'激活经络'或全面提升免疫系统。",
        "免疫系统的强弱受多种因素影响，包括遗传、年龄、睡眠质量、压力水平和整体营养状况，不可能通过单一饮品显著改变。",
        "某些病毒，如新冠病毒，是新型病原体，人体即使免疫系统强大也可能感染，饮食无法提供特异性保护。",
        "虽然生姜确实含有抗炎成分，红枣也富含维生素，但它们的效果是温和的，不可能创造'超级免疫力'。",
        "个人报告不感冒可能与多种因素有关，如减少暴露、良好的个人卫生习惯或者单纯的运气，不应完全归功于特定饮品。",
        "健康的免疫系统需要均衡饮食、充足睡眠、适当运动和管理压力，而非依赖所谓的'神奇配方'。"
    ],
    health_topics[4]: [  # 根治胃病话题
        "胃病包括多种疾病，如胃炎、胃溃疡、胃食管反流等，它们有不同的病因和治疗方法，不可能用一种简单技巧全部治愈。",
        "打嗝是胃部或食道中气体排出的自然反应，人为干预这个过程不会'重置'胃功能。",
        "慢性胃病通常需要药物治疗、饮食调整和生活方式改变的综合管理，而非简单的呼吸技巧。",
        "如果胃病症状持续存在，应该寻求专业医疗建议，延误治疗可能导致更严重的并发症。",
        "某些胃病与幽门螺杆菌感染或自身免疫反应有关，这些根本原因不可能通过控制打嗝来解决。",
        "虽然深呼吸技巧可能暂时缓解某些消化不适，但它不能替代适当的医疗诊断和治疗方案。"
    ]
}

# 对话生成器类
class DialogueGenerator:
    def __init__(self, api_key=None, weibo_users=None):
        self.client = ZhipuAI(api_key=api_key or "您的API密钥")
        self.weibo_users = weibo_users or []
    
    def select_topic(self, category=None, specific_topic=None):
        if specific_topic:
            return specific_topic
        
        if category:
            return random.choice(TOPIC_POOL[category])
        
        category = random.choice(list(TOPIC_POOL.keys()))
        return random.choice(TOPIC_POOL[category])
    
    def get_completion(self, messages):
        try:
            formatted_messages = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in messages
            ]
            
            response = self.client.chat.completions.create(
                model="glm-3-turbo",
                messages=formatted_messages,
                temperature=0.8
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API调用错误: {e}")
            time.sleep(1)
            try:
                response = self.client.chat.completions.create(
                    model="glm-3-turbo",
                    messages=formatted_messages,
                    temperature=0.8
                )
                return response.choices[0].message.content
            except:
                return "对不起，我现在无法回应，请稍后再试。"
    
    def select_weibo_user_as_patient(self):
        """从微博用户数据中选择一个作为患者角色"""
        if not self.weibo_users:
            return DialogueParticipant()  # 如果没有微博用户数据，创建随机患者
        
        # 选择一个随机的微博用户
        selected_user = random.choice(self.weibo_users)
        patient = DialogueParticipant(selected_user)
        patient.role = "患者"
        return patient
    
    def generate_short_response(self, prev_message):
        responses = [
            f"嗯", f"对", f"是的", f"好的", f"啊，对",
            prev_message.split("，")[0] + "，对的",
            f"确实", f"没错", f"嗯，确实", f"有道理",
            f"对对对", f"我也这么觉得", f"说得对"
        ]
        return random.choice(responses)
    
    def generate_medical_dialogue(self, medical_topic, turns=8, medical_examples=None):
        """使用微博用户作为患者角色生成医患对话"""
        # 创建医生和患者角色
        doctor = create_doctor_participant()
        patient = self.select_weibo_user_as_patient()  # 从微博用户中选择患者
        
        dialogue = []
        current_topic = medical_topic
        
        # 获取当前主题的支持和反对观点
        supporting_opinions = topic_opinions.get(current_topic, [])
        opposing_opinions = topic_counter_opinions.get(current_topic, [])
        
        # 决定医生和患者的观点立场（随机分配，确保有对立观点）
        patient_supports_topic = random.choice([True, False])
        doctor_supports_topic = not patient_supports_topic  # 医生采取相反立场
        
        # 患者先提问
        patient_query = self.generate_patient_query(patient, current_topic, medical_examples, 
                                                  supporting_opinions if patient_supports_topic else opposing_opinions)
        dialogue.append({"speaker": "患者", "content": patient_query})
        
        # 医生回应
        doctor_response = self.generate_doctor_response(doctor, [patient_query], current_topic,
                                                     supporting_opinions if doctor_supports_topic else opposing_opinions)
        dialogue.append({"speaker": "医生", "content": doctor_response})
        
        # 继续对话
        for i in range(turns - 2):
            if i % 2 == 0:
                # 患者提问或回应
                query = self.generate_patient_followup(patient, [msg["content"] for msg in dialogue], current_topic,
                                                     supporting_opinions if patient_supports_topic else opposing_opinions)
                dialogue.append({"speaker": "患者", "content": query})
            else:
                # 医生回应
                response = self.generate_doctor_response(doctor, [msg["content"] for msg in dialogue], current_topic,
                                                       supporting_opinions if doctor_supports_topic else opposing_opinions)
                dialogue.append({"speaker": "医生", "content": response})
        
        return {
            "topic": current_topic,
            "participants": [
                {
                    "role": "医生", 
                    "name": doctor.name,
                    "traits": doctor.traits, 
                    "education": doctor.education,
                    "supports_topic": doctor_supports_topic
                },
                {
                    "role": "患者", 
                    "name": patient.name,
                    "traits": patient.traits, 
                    "education": patient.education,
                    "user_id": patient.user_id,
                    "health_view": patient.health_view,
                    "description": patient.description,
                    "supports_topic": patient_supports_topic
                }
            ],
            "dialogue": dialogue
        }
    
    def generate_patient_query(self, patient, topic, medical_examples=None, opinion_references=None):
        example_text = ""
        if medical_examples and len(medical_examples) > 0:
            samples = random.sample(medical_examples, min(3, len(medical_examples)))
            example_text = "参考以下医疗问答示例（仅作为问题类型的参考，不要直接复制内容）：\n"
            for sample in samples:
                example_text += f"问：{sample['query']}\n"
                example_text += f"答：{sample['response']}\n\n"
        
        opinion_text = ""
        if opinion_references and len(opinion_references) > 0:
            opinion_text = "参考以下相关观点（可以融入到你的问题中，但不要直接照抄）：\n"
            selected_opinions = random.sample(opinion_references, min(3, len(opinion_references)))
            for opinion in selected_opinions:
                opinion_text += f"- {opinion}\n"
        
        system_prompt = (
            "你将模拟一个真实患者向医生咨询健康问题。请根据角色描述和健康话题，生成一个自然的问题。"
            "【重要】不要以'医生您好'、'尊敬的医生'等称呼开头，直接表达问题或困扰，模拟真实的面对面对话。"
            "这应该是一个真实患者会问的问题，包含具体症状描述或健康困扰，语言应该是口语化的。"
            "特别注意使用患者的健康观点和背景来塑造问题，如果患者有特定的健康理念（如偏好中医），应当体现出来。"
            "请适当融入提供的观点参考，让问题带有一定的立场倾向。"
        )
        
        user_prompt = f"""
        {patient.generate_description()}
        
        健康咨询话题: {topic}
        
        {example_text}
        
        {opinion_text}
        
        请生成一个患者向医生咨询的自然问题，应该：
        1. 包含具体症状或健康困扰的描述
        2. 使用日常口语而非医学专业术语
        3. 表达出患者的疑虑或担忧
        4. 可以包含患者的生活习惯或相关背景信息
        5. 反映患者的健康观点和个性特征
        6. 适当体现提供的观点参考中的一些看法
        
        只输出患者的问题，不要有任何解释或引导语。
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return self.get_completion(messages)
    
    def generate_patient_followup(self, patient, previous_messages, topic, opinion_references=None):
        """生成患者的后续回应，结合短回复和丰富回答"""
        # 短回复概率基础值
        short_response_probability = 0.3
        
        # 神经质高的人更容易给简短回复
        if patient.traits["神经质"] >= 5:
            short_response_probability += 0.1
        
        # 外向性低的人更倾向于简短回复
        if patient.traits["外向性"] <= 3:
            short_response_probability += 0.1
        
        # 如果上一条医生消息很长，提高简短回复概率
        doctor_message = previous_messages[-1]
        if len(doctor_message) > 100:
            short_response_probability += 0.1
        
        # 决定是否使用简短回复
        if random.random() < short_response_probability:
            return self.generate_short_response(doctor_message)
        
        # 准备观点参考内容
        opinion_text = ""
        if opinion_references and len(opinion_references) > 0:
            selected_opinions = random.sample(opinion_references, min(2, len(opinion_references)))
            opinion_text = "参考以下观点（仅作为立场参考，不要直接照抄）：\n"
            for opinion in selected_opinions:
                opinion_text += f"- {opinion}\n"
        
        # 如果不是短回复，有一定概率生成丰富的个性化回答
        rich_response_probability = 0.6
        if random.random() < rich_response_probability:
            # 生成丰富的个性化回答
            system_prompt = (
                "你将模拟一个真实患者在医疗咨询中的后续回应。请根据角色描述、对话历史和健康话题，生成一个自然的问题或回应。"
                "【重要】不要使用'尊敬的医生'、'医生您好'等称呼开头，直接进入主题，模拟真实的面对面对话。"
                "这应该是病人在医生回答后可能会提出的疑问、担忧或感谢，语言应口语化且符合患者的个性。"
                "要反映患者对该健康话题的立场和观点，与参考观点保持一致。"
            )
            
            dialogue_history = "\n".join([f"{'患者' if i%2==0 else '医生'}: {msg}" for i, msg in enumerate(previous_messages)])
            
            user_prompt = f"""
            {patient.generate_description()}
            
            健康咨询话题: {topic}
            
            {opinion_text}
            
            对话历史:
            {dialogue_history}
            
            请生成患者的后续回应，应该：
            1. 长度适中（100-200字），表达充分但不冗长
            2. 展现患者的性格特点（例如：开放性{patient.traits["开放性"]}/7、尽责性{patient.traits["尽责性"]}/7、外向性{patient.traits["外向性"]}/7等）
            3. 反映患者的健康观点和态度
            4. 包含个人经历或具体情况描述
            5. 可以提及患者背景中的相关信息
            6. 使用符合患者教育水平和说话风格的语言
            7. 融入提供的观点参考，保持立场一致性
            
            只输出患者的回应，不要有任何解释或引导语。
            """
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            return self.get_completion(messages)
        
        # 普通回应（不是短回复也不是特别丰富的回答）
        system_prompt = (
            "你将模拟一个真实患者在医疗咨询中的后续回应。请根据角色描述、对话历史和健康话题，生成一个自然的问题或回应。"
            "【重要】不要使用'尊敬的医生'、'医生您好'等称呼开头，直接进入主题，模拟真实的面对面对话。"
            "这应该是病人在医生回答后可能会提出的疑问、担忧或感谢，语言应口语化且符合患者的个性。"
            "要反映患者对该健康话题的立场和观点，与参考观点保持一致。"
        )
        
        dialogue_history = "\n".join([f"{'患者' if i%2==0 else '医生'}: {msg}" for i, msg in enumerate(previous_messages)])
        
        user_prompt = f"""
        {patient.generate_description()}
        
        健康咨询话题: {topic}
        
        {opinion_text}
        
        对话历史:
        {dialogue_history}
        
        请生成患者的后续回应，应该是：
        1. 对医生解释的进一步疑问
        2. 提供更多症状或背景信息
        3. 对医生建议的顾虑或考虑
        4. 表达与患者健康观点相关的看法
        5. 坚持或质疑自己的立场，但总体保持与提供的观点参考一致
        
        只输出患者的问题或回应，不要有任何解释或引导语。
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return self.get_completion(messages)
    
    def generate_doctor_response(self, doctor, previous_messages, topic, opinion_references=None):
        # 决定是否使用简短回复
        short_response_probability = 0.05  # 医生使用简短回复的概率较低
        
        # 如果上一条患者消息很短，提高简短回复概率
        patient_message = previous_messages[-1]
        if len(patient_message) < 15:
            short_response_probability += 0.1
        
        # 决定是否使用简短回复
        if random.random() < short_response_probability:
            return self.generate_short_response(patient_message)
        
        # 准备观点参考内容
        opinion_text = ""
        if opinion_references and len(opinion_references) > 0:
            selected_opinions = random.sample(opinion_references, min(2, len(opinion_references)))
            opinion_text = "参考以下观点（仅作为立场参考，不要直接照抄）：\n"
            for opinion in selected_opinions:
                opinion_text += f"- {opinion}\n"
        
        system_prompt = (
            "你将模拟一个专业医生回答患者的健康咨询。请根据医生角色描述、对话历史和健康话题，生成一个专业且温和的回应。"
            "回应应该专业但不使用过多专业术语，尽量用患者容易理解的方式解释医学概念，并表现出关心和同理心。"
            "【重要】不要使用'患者先生/女士'、'尊敬的患者'等称呼开头，直接进入主题，模拟真实的面对面对话。"
            "如果患者提到特定健康观点（如中医理念），医生应尊重但在必要时给予科学引导。"
            "医生的回应应该体现提供的观点参考立场，但表达方式应专业、谨慎。"
        )
        
        dialogue_history = "\n".join([f"{'患者' if i%2==0 else '医生'}: {msg}" for i, msg in enumerate(previous_messages)])
        
        user_prompt = f"""
        {doctor.generate_description()}
        
        健康咨询话题: {topic}
        
        {opinion_text}
        
        对话历史:
        {dialogue_history}
        
        请生成医生的专业回应，应该：
        1. 针对患者的问题提供专业且准确的医疗信息
        2. 解释医学概念时使用患者容易理解的语言
        3. 表现出关心和尊重，不轻视患者的疑虑
        4. 在合适的情况下提供健康建议或下一步的诊疗方向
        5. 语气保持专业但友善
        6. 适当体现参考观点中的立场，但表达方式应专业、谨慎
        
        只输出医生的回应，不要有任何解释或引导语。
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return self.get_completion(messages)
    
    def save_dialogue(self, dialogue, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            for turn in dialogue["dialogue"]:
                f.write(f"{turn['speaker']}: {turn['content']}\n")
        print(f"对话已保存到 {filename}")

    def save_dialogue_with_metadata(self, dialogue, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dialogue, f, ensure_ascii=False, indent=2)
        print(f"带元数据的对话已保存到 {filename}")

# 工具函数
def load_medical_dataset(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    medical_examples = []
    for item in data:
        if "instruct" in item and "output" in item:
            medical_examples.append({
                "query": item["instruct"],
                "response": item["output"]
            })
    
    return medical_examples

def enhance_dialogue_diversity(generator, dialogue, diversity_level=0.3):
    enhanced_dialogue = dialogue.copy()
    dialogue_content = enhanced_dialogue["dialogue"]
    
    # 处理需要丰富的短回复
    for idx, turn in enumerate(dialogue_content):
        if turn.get("needs_enrichment", False) and turn["speaker"] == "患者":
            # 找到这个患者的信息
            patient_info = None
            for participant in enhanced_dialogue["participants"]:
                if participant["role"] == "患者":
                    patient_info = participant
                    break
            
            if not patient_info:
                continue
                
            # 创建临时患者对象用于生成回答
            temp_patient = DialogueParticipant()
            temp_patient.traits = patient_info["traits"]
            temp_patient.health_view = patient_info.get("health_view", "")
            temp_patient.name = patient_info["name"]
            temp_patient.education = patient_info["education"]
            
            # 获取对话历史
            history_until_now = [msg["content"] for msg in dialogue_content[:idx]]
            
            # 获取该主题的相关观点
            current_topic = enhanced_dialogue["topic"]
            opinions = topic_opinions.get(current_topic, []) if patient_info.get("supports_topic", True) else topic_counter_opinions.get(current_topic, [])
            
            # 重新生成丰富的回答
            system_prompt = (
                "你将为患者创建一个丰富的回应，替换原本的简短回复。"
                "回答应该完全展现患者的个性特征和健康观点，同时保持自然流畅。"
                "回答应该与患者的立场保持一致，体现对健康话题的特定观点。"
            )
            
            user_prompt = f"""
            患者简介：
            - 名字：{temp_patient.name}
            - 教育程度：{temp_patient.education}
            - 开放性：{patient_info["traits"]["开放性"]}/7
            - 尽责性：{patient_info["traits"]["尽责性"]}/7
            - 外向性：{patient_info["traits"]["外向性"]}/7
            - 宜人性：{patient_info["traits"]["宜人性"]}/7
            - 神经质：{patient_info["traits"]["神经质"]}/7
            - 健康观点：{temp_patient.health_view}
            
            对话历史上下文：
            {history_until_now[-1] if history_until_now else "对话开始"}
            
            原短回复：
            {turn["content"]}
            
            健康主题：
            {current_topic}
            
            参考观点（仅作为立场参考，不要直接照抄）：
            {random.sample(opinions, min(2, len(opinions))) if opinions else "无特定观点参考"}
            
            请创建一个丰富的回应，应该：
            1. 完全反映患者的性格特点和健康观点
            2. 长度适中（100-200字），表达充分
            3. 包含个人经历或生活细节
            4. 展现患者对健康的态度和关注点
            5. 使用符合患者特质的语言风格
            6. 与患者的立场保持一致
            
            只输出患者的新回应，不要有任何解释。
            """
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            enhanced_content = generator.get_completion(messages)
            dialogue_content[idx]["content"] = enhanced_content
            if "needs_enrichment" in dialogue_content[idx]:
                del dialogue_content[idx]["needs_enrichment"]
    
    return enhanced_dialogue

def post_process_dialogue(dialogue, max_filler_ratio=0.5):
    dialogue_content = dialogue["dialogue"]
    
    # 计算以语气词开头的句子数量
    filler_starters = ["哎", "啊", "嗯", "呃", "哦", "唉", "哼", "喂", "嘿"]
    filler_starts = 0
    for turn in dialogue_content:
        content = turn["content"]
        if any(content.startswith(filler) for filler in filler_starters):
            filler_starts += 1
    
    current_ratio = filler_starts / len(dialogue_content)
    
    # 如果超过了最大比例，随机移除一些语气词
    if current_ratio > max_filler_ratio:
        remove_count = int((current_ratio - max_filler_ratio) * len(dialogue_content))
        turns_with_fillers = [i for i, turn in enumerate(dialogue_content) 
                            if any(turn["content"].startswith(filler) for filler in filler_starters)]
        
        if turns_with_fillers:
            to_remove = random.sample(turns_with_fillers, min(remove_count, len(turns_with_fillers)))
            
            for idx in to_remove:
                content = dialogue_content[idx]["content"]
                for filler in filler_starters:
                    if content.startswith(filler):
                        dialogue_content[idx]["content"] = content[len(filler):].lstrip('，, ')
                        break
    
    # 确保短回复的比例不要过高
    short_replies = 0
    for turn in dialogue_content:
        if len(turn["content"]) < 15:  # 15个字符以内的回复视为短回复
            short_replies += 1
    
    current_short_ratio = short_replies / len(dialogue_content)
    
    # 患者短回复比例控制在20-30%之间
    min_short_ratio = 0.2
    max_short_ratio = 0.3
    
    # 如果短回复比例过高，随机将一些短回复转换为标准回复
    if current_short_ratio > max_short_ratio:
        reduce_count = int((current_short_ratio - max_short_ratio) * len(dialogue_content))
        short_patient_turns = [i for i, turn in enumerate(dialogue_content) 
                            if len(turn["content"]) < 15 and turn["speaker"] == "患者"]
        
        if short_patient_turns and reduce_count > 0:
            to_extend = random.sample(short_patient_turns, min(reduce_count, len(short_patient_turns)))
            
            for idx in to_extend:
                # 这些短回复将在enhance_dialogue_diversity函数中被替换为更丰富的内容
                # 添加标记，后续处理
                dialogue_content[idx]["needs_enrichment"] = True
    
    return dialogue

def generate_medical_dataset(num_dialogues=20, output_dir="medical_dialogues", api_key=None, weibo_csv="weibo_users_extended.csv", medical_json="CMtMedQA_test.json"):
    """
    生成医疗健康对话数据集，使用真实微博用户数据作为患者背景
    
    参数:
    - num_dialogues: 要生成的对话数量
    - output_dir: 输出目录
    - api_key: 智谱API密钥
    - weibo_csv: 微博用户数据CSV文件路径
    - medical_json: 医疗问答JSON文件路径
    """
    import os
    import time
    from datetime import datetime
    
    # 创建输出目录，使用日期时间作为子目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir, f"dataset_{timestamp}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 加载微博用户数据
    print(f"正在加载微博用户数据...")
    weibo_users = load_weibo_users(weibo_csv)
    print(f"成功加载 {len(weibo_users)} 个用户数据")
    
    # 加载医疗问答数据
    print(f"正在加载医疗问答数据...")
    medical_examples = load_medical_dataset(medical_json)
    print(f"成功加载 {len(medical_examples)} 条医疗问答")
    
    # 创建生成器
    generator = DialogueGenerator(api_key, weibo_users)
    
    # 生成统计信息文件
    stats_file = os.path.join(output_dir, "dataset_stats.txt")
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write(f"生成时间: {timestamp}\n")
        f.write(f"对话数量: {num_dialogues}\n")
        f.write(f"用户数据源: {weibo_csv}\n")
        f.write(f"医疗问答源: {medical_json}\n\n")
        f.write("虚假健康信息主题:\n")
        for topic in health_topics:
            f.write(f"- {topic}\n")
        f.write("\n对话详情:\n")
    
    # 记录开始时间
    start_time = time.time()
    
    # 用来记录已使用的微博用户ID，确保不重复
    used_user_ids = set()
    
    # 生成多个对话
    for i in range(num_dialogues):
        try:
            # 随机选择一个虚假健康信息主题
            topic = random.choice(health_topics)
            turns = random.randint(6, 10)  # 医疗对话通常不会太长
            
            print(f"正在生成医疗对话 {i+1}/{num_dialogues}，话题: {topic}，轮数: {turns}")
            
            # 生成医患对话
            dialogue = generator.generate_medical_dialogue(
                topic,
                turns=turns,
                medical_examples=random.sample(medical_examples, min(5, len(medical_examples)))
            )
            
            # 记录使用的用户ID
            patient_info = dialogue["participants"][1]  # 患者信息
            used_user_ids.add(patient_info["user_id"])
            
            # 后处理对话，调整语气词使用
            dialogue = post_process_dialogue(dialogue, max_filler_ratio=0.3)
            
            # 增强对话多样性
            enhanced_dialogue = enhance_dialogue_diversity(generator, dialogue)
            
            # 创建对话特定目录
            dialogue_dir = os.path.join(output_dir, f"dialogue_{i+1}")
            os.makedirs(dialogue_dir, exist_ok=True)
            
            # 保存对话
            generator.save_dialogue(
                enhanced_dialogue, 
                os.path.join(dialogue_dir, "dialogue.txt")
            )
            generator.save_dialogue_with_metadata(
                enhanced_dialogue, 
                os.path.join(dialogue_dir, "metadata.json")
            )
            
            # 更新统计信息
            with open(stats_file, 'a', encoding='utf-8') as f:
                f.write(f"\n对话 {i+1}:\n")
                f.write(f"  话题: {topic}\n")
                f.write(f"  轮数: {turns}\n")
                f.write(f"  患者: {patient_info['name']} (用户ID: {patient_info['user_id']})\n")
                f.write(f"  患者立场: {'支持' if patient_info.get('supports_topic', False) else '反对'}\n")
                f.write(f"  患者健康观点: {patient_info['health_view'][:50]}...\n" if len(patient_info['health_view']) > 50 else f"  患者健康观点: {patient_info['health_view']}\n")
            
            # 每生成5个对话后暂停一下，避免API调用频率过高
            if (i + 1) % 5 == 0 and i < num_dialogues - 1:
                print(f"已完成 {i+1}/{num_dialogues} 个对话，暂停30秒...")
                time.sleep(30)
                
        except Exception as e:
            print(f"生成对话 {i+1} 时出错: {e}")
            with open(os.path.join(output_dir, "errors.log"), 'a', encoding='utf-8') as f:
                f.write(f"对话 {i+1} 生成失败: {str(e)}\n")
            time.sleep(10)  # 出错后暂停一下再继续
    
    # 记录结束时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # 更新统计信息
    with open(stats_file, 'a', encoding='utf-8') as f:
        f.write(f"\n总耗时: {elapsed_time:.2f} 秒\n")
        f.write(f"平均每个对话耗时: {elapsed_time/num_dialogues:.2f} 秒\n")
        f.write(f"使用的不同用户数: {len(used_user_ids)}\n")
    
    print(f"\n医疗对话数据集生成完成!")
    print(f"总共生成了 {num_dialogues} 个医患对话")
    print(f"使用了 {len(used_user_ids)} 个不同的微博用户作为患者角色")
    print(f"数据保存在: {output_dir}")
    print(f"总耗时: {elapsed_time:.2f} 秒")

# 主程序
if __name__ == "__main__":
    api_key = "2fbfc2acf9e54a09886d422966fa3448.lss3vj7BdP327Wdh"  # 替换为您的密钥
    
    # 生成医疗对话数据集
    generate_medical_dataset(
        num_dialogues=20,
        output_dir="medical_dialogues",
        api_key=api_key,
        weibo_csv="weibo_users_extended.csv",
        medical_json="CMtMedQA_test.json"
    )