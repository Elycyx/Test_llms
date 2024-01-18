from transformers import pipeline
import time

# 创建文本生成管道
generator = pipeline('text-generation', model="facebook/opt-350m")

question = "What is the capital of USA?"
start_time = time.time()
# 生成文本，同时指定一些参数来定制生成过程
generated_texts = generator(question)
end_time = time.time()
time_spent = end_time - start_time

# 提取生成的文本
generated_text = generated_texts[0]["generated_text"]

# 定位到问题的末尾
question_end_index = generated_text.find(question) + len(question)

# 提取模型的回答
answer = generated_text[question_end_index:]

# 计算速度
tokens = generator.tokenizer(answer, return_tensors="pt")
num_tokens = tokens.input_ids.size(1)
speed = num_tokens / time_spent

# 打印回答和速度
print(answer.strip())  # 使用strip()来移除前后的空格和换行
print('speed: '+ str(speed) + ' tokens per second')
