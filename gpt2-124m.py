# Install transformers from source - only needed for versions <= v4.34
# pip install git+https://github.com/huggingface/transformers.git
# pip install accelerate

import torch
from transformers import pipeline
import time

pipe = pipeline("text-generation", model="gpt2", torch_dtype=torch.bfloat16, device_map="auto")

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
outputs = pipe(prompt, max_new_tokens=256, do_sample=True, top_k=50, top_p=0.95)
end_time = time.time()
time_spent = end_time - start_time
print(outputs[0]["generated_text"])
tokens = pipe.tokenizer(outputs[0]["generated_text"], return_tensors="pt")
num_tokens = tokens.input_ids.size(1)
speed = num_tokens / time_spent
print('speed: '+ str(speed) + ' tokens per second')

