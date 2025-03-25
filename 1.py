from openai import OpenAI
from tqdm import tqdm
import time

# 配置客户端
openai_api_key = "ANY THING"
openai_api_base = "http://xunziallm.njau.edu.cn:21180/v1"

# 添加重试逻辑
max_retries = 3
retry_delay = 2

for attempt in range(max_retries):
    try:
        client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        
        for i in tqdm(range(0,1)):
            chat_response = client.chat.completions.create(
                model="/home/gpu0/xunzi_web/Xunzi-Qwen1.5-7B_chat",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": '根据提供的文本，按照关系scheme组合(人物, PO/官職, 官職),(人物, PP/態度傾向/消極, 人物),(人物, PL/其他, 地点),(人物, PL/居, 地点),(人物代词, 態度傾向/消極, 人物)抽取出符合描述的关系三元组\n奏上，上令公卿列侯宗室集議，莫敢難，獨竇嬰爭之，由此與錯有卻。'},
                ]
            )
            print(chat_response.choices[0].message.content)
        break
    except Exception as e:
        print(f"Attempt {attempt + 1} failed. Retrying in {retry_delay} seconds...")
        time.sleep(retry_delay)