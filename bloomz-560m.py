# pip install -q transformers accelerate
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

checkpoint = "bigscience/bloomz-560m"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")
start_time = time.time()
inputs = tokenizer.encode("What is the capital of USA?", return_tensors="pt").to("cuda")
outputs = model.generate(inputs)
end_time = time.time()
time_spent = end_time - start_time
num_tokens = outputs.size(1)
speed = num_tokens / time_spent
print(tokenizer.decode(outputs[0]))
print('speed: '+ str(speed) + ' tokens per second')