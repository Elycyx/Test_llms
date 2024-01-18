from transformers import pipeline
import torch
import time

generate_text = pipeline(model="aisquared/dlite-v2-774m", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
start_time = time.time()
res = generate_text("What is the capital of USA?")
end_time = time.time()
time_spent = end_time - start_time
tokens = generate_text.tokenizer(res, return_tensors="pt")
num_tokens = tokens.input_ids.size(1)
speed = num_tokens / time_spent
print(res)
print('speed: '+ str(speed) + ' tokens per second')