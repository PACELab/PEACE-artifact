import openai
import time
import csv
import transformers
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool

import sys

##
# start vllm server - docker run --gpus all -it --rm --ipc=host nba556677/chatbot:vllm python -m vllm.entrypoints.openai.api_server --model facebook/opt-6.7b --gpu-memory-utilization 0.75
##

openai.api_key = "EMPTY" # Not support yet
openai.api_base = f"http://{sys.argv[1]}:8000/v1"

model = "mistralai/Mistral-7B-Instruct-v0.2"
texts= f"""In the vast expanse of the universe, where countless galaxies dance in a cosmic symphony, the wonders of existence unfold before our eyes. From the smallest particles that weave the tapestry of reality to the majestic celestial bodies that adorn the night sky, every corner of this magnificent cosmos holds secrets waiting to be discovered. Through the lens of science, humanity has embarked on an awe-inspiring journey of exploration and understanding, unraveling the mysteries of life, the universe, and everything in between. Throughout history, we have witnessed remarkable leaps in knowledge and technology, propelling us forward in our quest for discovery. From the ancient philosophers who pondered the nature of existence to the Renaissance thinkers who ushered in an era of scientific enlightenment, each generation has built upon the knowledge of its predecessors, pushing the boundaries of human understanding ever further. They laid the foundation for the scientific revolution, which sparked an insatiable curiosity to explore the unknown. In the 21st century, humanity finds itself at the precipice of unprecedented advancements. Breakthroughs in fields such as artificial intelligence, biotechnology, and space exploration have the potential to reshape our world in profound ways. We stand on the cusp of unlocking the mysteries of the human brain, decoding the building blocks of life itself, and venturing beyond our planetary borders. With each passing day, scientists and innovators around the globe strive to unlock the secrets of our universe, harnessing the power of technology to tackle the grand challenges that lie ahead. Space agencies and private companies collaborate to send missions to distant planets, seeking clues about our origins and the possibility of extraterrestrial life. We send probes and telescopes deeper into space, capturing breathtaking images and collecting data that expands our understanding of the cosmos. Yet, amidst the rapid progress and scientific marvels, we must not forget the importance of preserving our fragile planet and fostering a harmonious coexistence with the natural world. Climate change, habitat destruction, and pollution threaten the very ecosystems that sustain life on Earth. It is our collective responsibility to safeguard this precious home and seek sustainable solutions for a better future. Moreover, as we navigate the complexities of a globalized world, it is crucial to embrace diversity, empathy, and inclusivity. Our differences in culture, religion, and perspective should be celebrated as strengths that enrich our collective human experience.  By fostering collaboration and understanding, we can overcome the challenges that divide us, forging a path towards a more compassionate and united global society. In the realms of art, literature, and music, human creativity continues to flourish, serving as a testament to our capacity for imagination and expression. Through masterful strokes of a paintbrush, captivating words on a page, or melodies that resonate with our souls, artists shape our understanding of the world, evoke emotions, and challenge our perceptions. They give voice to the human experience and inspire us to see the beauty in everyday life. Art reminds us of our shared humanity and the power of beauty to transcend the boundaries of language and culture. It connects us at a deeper level, allowing us to explore the depths of our emotions and ignite a sense of wonder within us. In galleries, theaters, and concert halls, we are transported to different worlds and given glimpses into the minds of those who create. In the digital age, the rapid expansion of information and communication technologies has transformed the way we connect, learn, and interact. Social media platforms, virtual reality, and the internet have opened up new avenues for global dialogue and collaboration. The world has become more interconnected than ever before, bridging distances and enabling people from different corners of the globe to exchange ideas and cultures instantaneously. As we gaze into the future, the possibilities that lie ahead are both exhilarating and humbling. From the exploration of distant planets to the potential for groundbreaking scientific discoveries, humanity's journey of discovery and self-discovery continues to unfold. It is a reminder that we are part of something greater, a cosmic tapestry of existence, bound together by the shared wonder and curiosity that defines us as human beings. So, let us embark on this grand adventure together,"""

prompt = f"Translate the ```{texts}``` to French"
#prompt = f"The woman worked as a"
n_threads = int(sys.argv[2])
index = np.arange(0,n_threads, 1).tolist()






#prompt = f"What is the capital of France?"
# create a completion
#completion = openai.Completion.create(model=model, prompt=prompt, max_tokens=1024)
# print the completion
#print(completion.choices[0].text)

# create a chat completion
t_start = time.time()

pool = ThreadPool(n_threads)

def request(index):
    start = time.time()
    completion = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens ="500"
    )
    # print the completion
#t_end = time.time() - t_start
    print(completion.choices[0].message.content)
    print(f"request #{index} finishes in {time.time()-start}s")
    #print(completion)
#t = transformers.AutoTokenizer.from_pretrained('/workspace/FastChat/vicuna-13b-v1.3/', use_fast=False)
#output_token = t.tokenize(text=completion.choices[0].message.content)
results = pool.map(request, index)
pool.close()
pool.join()
#countOfWords = len(completion.choices[0].message.content.split())
#countOfWords = len(output_token)
t_end = time.time() - t_start
#print(countOfWords)
#count = [countOfWords]

#with open('./output_length.csv', 'a',newline='') as f:
#    writer = csv.writer(f)
#    writer.writerow(count)
print("duration:", t_end)

