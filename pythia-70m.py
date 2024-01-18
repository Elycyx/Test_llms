from transformers import GPTNeoXForCausalLM, AutoTokenizer
import time

model = GPTNeoXForCausalLM.from_pretrained(
  "EleutherAI/pythia-70m-deduped",
  revision="step3000",
  cache_dir="./pythia-70m-deduped/step3000",
)

tokenizer = AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-70m-deduped",
  revision="step3000",
  cache_dir="./pythia-70m-deduped/step3000",
)
start_time = time.time()
inputs = tokenizer("Hello, I am ", return_tensors="pt")
tokens = model.generate(**inputs)
end_time = time.time()
time_spent = end_time - start_time
num_tokens = tokens.size(1)
speed = num_tokens / time_spent
print(tokenizer.decode(tokens[0]))
print('speed: '+ str(speed) + ' tokens per second')