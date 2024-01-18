import torch
from transformers import AutoTokenizer, pipeline
import time

model = 'ericzzz/falcon-rw-1b-instruct-openorca'

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = pipeline(
   'text-generation',
   model=model,
   tokenizer=tokenizer,
   torch_dtype=torch.bfloat16,
   device_map='auto',
)

with open('1.txt', 'r') as f:
   pr = f.read()

system_message = 'You are a helpful assistant. Give short answers.'
instruction = 'What is the capital of USA?'
# system_message = pr
# instruction = 'Go to Room 209.'
start_time = time.time()
prompt = f'<SYS> {system_message} <INST> {instruction} <RESP> '

response = pipeline(
   prompt, 
   max_length=2000,
   repetition_penalty=1.05
)
end_time = time.time()
time_spent = end_time - start_time

# 提取回答部分
answer_start_idx = response[0]['generated_text'].find('<RESP>') + len('<RESP> ')
answer = response[0]['generated_text'][answer_start_idx:].strip()
tokens = tokenizer(answer, return_tensors="pt")
num_tokens = tokens.input_ids.size(1)
speed = num_tokens / time_spent


print(answer)
print('speed: '+ str(speed) + ' tokens per second')
