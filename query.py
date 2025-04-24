import json

def convert_dialogues(file_path, output_file):
    # 读取JSON文件
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 打开输出文件
    with open(output_file, 'w', encoding='utf-8') as out_f:
        # 遍历所有对话
        for dialogue in data['LLM_generated_dialogues']:
            # 获取对话ID
            dialogue_id = dialogue['id']
            
            # 遍历对话中的每轮对话
            for i, round_dialogue in enumerate(dialogue['rounds']):
                # 每轮对话包含两个元素：患者问题和医生回答
                patient_query = round_dialogue[0]
                doctor_response = round_dialogue[1]
                
                # 按照要求的格式写入文件
                out_f.write(f"患者：{patient_query}\n")
                out_f.write(f"医生：{doctor_response}\n\n")
            
            # 在每个对话结束后添加原序号
            out_f.write(f"原序号: {dialogue_id}\n\n")

        for dialogue in data['human_generated_dialogues']:
            # 获取对话ID
            dialogue_id = dialogue['id']
            
            # 遍历对话中的每轮对话
            for i, round_dialogue in enumerate(dialogue['rounds']):
                # 每轮对话包含两个元素：患者问题和医生回答
                patient_query = round_dialogue[0]
                doctor_response = round_dialogue[1]
                
                # 按照要求的格式写入文件
                out_f.write(f"患者：{patient_query}\n")
                out_f.write(f"医生：{doctor_response}\n\n")
            
            # 在每个对话结束后添加原序号
            out_f.write(f"原序号: {dialogue_id}\n\n")

# 调用函数
convert_dialogues('dialogues.json', 'formatted_dialogues.txt')