import json
import os
import csv
import time
from datetime import datetime
from openai import OpenAI

def read_dialogues(file_path):
    """读取对话数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 合并人类生成和LLM生成的对话
    all_dialogues = []
    
    # 处理人类生成的对话
    for dialogue in data.get("human_generated_dialogues", []):
        all_dialogues.append({
            "id": dialogue["id"],
            "rounds": dialogue["rounds"],
            "type": "human"
        })
    
    # 处理LLM生成的对话
    for dialogue in data.get("LLM_generated_dialogues", []):
        all_dialogues.append({
            "id": dialogue["id"],
            "rounds": dialogue["rounds"],
            "type": "llm"
        })
    
    return all_dialogues

def format_dialogue_for_rating(dialogue):
    """将对话格式化为发送给LLM的文本"""
    dialogue_text = f"对话ID: {dialogue['id']}\n\n"
    
    for i, round_data in enumerate(dialogue["rounds"]):
        dialogue_text += f"轮次 {i+1}:\n"
        dialogue_text += f"患者: {round_data[0]}\n"
        if len(round_data) > 1:
            dialogue_text += f"医生: {round_data[1]}\n"
        dialogue_text += "\n"
    
    return dialogue_text

def create_prompt(dialogue_text):
    """创建发送给LLM的提示"""
    prompt = f"""请你对下面这段医患对话进行打分，1-5分，越高代表你越认可题目的陈述，题目为：
A. 自然流畅度:对话是否流畅自然，是否存在不合理的跳转或生硬的表达
B. 内容真实性:内容是否符合真实医患交流的特点，包括问题提出、解答方式等
C. 角色一致性:发言是否与角色身份（医生/患者）相符，角色是否在对话中保持一致
D. 专业可信度:医生回答的专业性和可信度，患者问题的合理性和真实性
E. 情感自然度:对话中表达的情感是否自然，是否有适当的情绪波动
F. 总体真实感:该对话是否属于真实对话而非AI生成对话，并标明信心程度

重要提示：
- 对每个维度独立评分，不要给所有维度相同的分数
- 每个维度都有各自的特点，请分别考量
- 特别是F维度，请严格评估对话是否感觉像真实人类对话，AI生成的对话通常分数会低一些
- 仔细思考后再给出评分，不同维度分数应有区别

你的输出结果必须仅为标准的CSV格式，即:
ID,A,B,C,D,E,F
[对话ID],[自然流畅度分数],[内容真实性分数],[角色一致性分数],[专业可信度分数],[情感自然度分数],[总体真实感分数]

请不要解释你的评分理由，只需返回上述格式的评分结果。

以下是需要评分的对话:

{dialogue_text}
"""
    return prompt

def get_deepseek_client(api_key):
    """获取DeepSeek API客户端"""
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

def rate_dialogue_with_deepseek(client, dialogue, model_name="deepseek-chat", temperature=0.7):
    """使用DeepSeek对对话进行评分"""
    dialogue_text = format_dialogue_for_rating(dialogue)
    prompt = create_prompt(dialogue_text)
    
    try:
        # 调用DeepSeek API
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "你是一个专业的医患对话评估专家。"},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            stream=False
        )
        
        # 解析结果
        rating_text = response.choices[0].message.content.strip()
        
        # 提取评分行
        lines = rating_text.split('\n')
        for line in lines:
            if dialogue["id"] in line:
                return line
        
        # 如果没有找到对应ID的行，返回完整响应
        return rating_text
    
    except Exception as e:
        print(f"对话 {dialogue['id']} 评分失败: {e}")
        return None

def save_ratings_to_csv(ratings, output_file):
    """将评分结果保存到CSV文件"""
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        f.write("ID,A,B,C,D,E,F\n")  # 写入表头
        for rating in ratings:
            if rating:
                f.write(f"{rating}\n")

def main(json_file_path, output_dir, api_key, model_name="deepseek-chat", batch_size=5):
    """主函数"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置输出文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"dialogue_ratings_deepseek_{timestamp}.csv")
    
    # 读取对话数据
    dialogues = read_dialogues(json_file_path)
    print(f"读取了 {len(dialogues)} 段对话")
    
    # 获取DeepSeek客户端
    client = get_deepseek_client(api_key)
    
    # 对对话进行评分
    ratings = []
    for i, dialogue in enumerate(dialogues):
        print(f"正在评分对话 {dialogue['id']} ({i+1}/{len(dialogues)})")
        rating = rate_dialogue_with_deepseek(client, dialogue, model_name)
        if rating:
            ratings.append(rating)
            print(f"评分结果: {rating}")
        
        # 每处理一批次，保存一次结果
        if (i + 1) % batch_size == 0 or i == len(dialogues) - 1:
            save_ratings_to_csv(ratings, output_file)
            print(f"已将评分结果保存到 {output_file}")
        
        # 添加延迟，避免API调用过于频繁
        if i < len(dialogues) - 1:
            time.sleep(1)
    
    print(f"完成! 共评分 {len(ratings)} 段对话")

if __name__ == "__main__":
    # 配置参数
    json_file_path = "dialogues.json"  # 对话数据文件路径
    output_dir = "ratings_results"     # 输出目录
    
    # DeepSeek API配置
    api_key = "sk-dbe63fa766654958994627ad286a6f3f"  # 请替换为你的DeepSeek API密钥
    model_name = "deepseek-chat"       # DeepSeek模型名称
    
    # 运行主函数
    main(json_file_path, output_dir, api_key, model_name)