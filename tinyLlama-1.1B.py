# Install transformers from source - only needed for versions <= v4.34
# pip install git+https://github.com/huggingface/transformers.git
# pip install accelerate

import torch
from transformers import pipeline
import time

pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

# We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a pirate",
    },
    {"role": "user", "content": "What is the capital of USA?"},
]
start_time = time.time()
prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipe(prompt, max_new_tokens=256, do_sample=False, top_k=50, top_p=0.95)
end_time = time.time()

# 提取生成的文本
generated_text = outputs[0]["generated_text"]

# 定位到用户问题的末尾
question_end_index = generated_text.find(messages[-1]["content"]) + len(messages[-1]["content"])

# 提取模型的回答
answer = generated_text[question_end_index:]

# 打印回答
print(answer.strip())

time_spent = end_time - start_time
tokens = pipe.tokenizer(answer, return_tensors="pt")
num_tokens = tokens.input_ids.size(1)
speed = num_tokens / time_spent
print('speed: '+ str(speed) + ' tokens per second')

