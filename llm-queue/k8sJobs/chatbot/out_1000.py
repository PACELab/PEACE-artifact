import openai
import time
import csv
import transformers
import sys 

openai.api_key = "EMPTY" # Not support yet
openai.api_base = f"http://{sys.argv[1]}:8000/v1"

model = "facebook/opt-6.7b"
#texts = f"""In the vast expanse of the universe, where countless galaxies dance in a cosmic symphony, the wonders of existence unfold before our eyes. From the smallest particles that weave the tapestry of reality to the majestic celestial bodies that adorn the night sky, every corner of this magnificent cosmos holds secrets waiting to be discovered. Through the lens of science, humanity has embarked on an awe-inspiring journey of exploration and understanding, unraveling the mysteries of life, the universe, and everything in between. Throughout history, we have witnessed remarkable leaps in knowledge and technology, propelling us forward in our quest for discovery."""
#prompt = f"Translate the ```{texts}``` to French"
#prompt = f"Introduce the universe"
prompt = f"Introduce the universe in 1000 words."
# create a completion
#completion = openai.Completion.create(model=model, prompt=prompt, max_tokens=1024)
# print the completion
#print(completion.choices[0].text)

# create a chat completion
t_start = time.time()
completion = openai.ChatCompletion.create(
  model=model,
  messages=[{"role": "user", "content": prompt}],
  temperature=0,
  max_tokens ="1000"
)
# print the completion
t_end = time.time() - t_start
print(completion.choices[0].message.content)
print(completion)
t = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)
output_token = t.tokenize(text=completion.choices[0].message.content)

#countOfWords = len(completion.choices[0].message.content.split())
countOfWords = len(output_token)

print(countOfWords)
count = [countOfWords]

#with open('./output_length.csv', 'a',newline='') as f:
#    writer = csv.writer(f)
#    writer.writerow(count)
print("duration:", t_end)