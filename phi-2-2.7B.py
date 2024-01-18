import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# torch.cuda.set_device(0)

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype=torch.float16, device_map="cuda", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

start_time = time.time()
inputs = tokenizer('''What is the capital of USA?''', return_tensors="pt", return_attention_mask=False)
inputs = inputs.to(torch.device("cuda"))
outputs = model.generate(**inputs, max_length=1000)
end_time = time.time()
time_spent = end_time - start_time
text = tokenizer.batch_decode(outputs)[0]
tokens = tokenizer(text, return_tensors="pt")
num_tokens = tokens.input_ids.size(1)
speed = num_tokens / time_spent
print(text)
print('speed: '+ str(speed) + ' tokens per second')