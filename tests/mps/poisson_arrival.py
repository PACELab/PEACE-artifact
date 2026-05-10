import time
import random
import openai
import threading
import numpy as np
import argparse
import json
import os
from datetime import datetime

##
# start vllm server - docker run --gpus all -it --rm --ipc=host nba556677/chatbot:vllm python -m vllm.entrypoints.openai.api_server --model facebook/opt-6.7b --gpu-memory-utilization 0.75
##
import sys
def request(index, ip, port, model):
    start = time.time()
    openai.api_base = f"http://{ip}:{port}/v1"
    completion = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens="500"
    )
    print(completion.choices[0].message.content)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] request #{index} arrivaltime : {datetime.fromtimestamp(start)} waiting+processtime: {time.time()-start}s")
    sys.stdout.flush()

def load_interarrival_times(file_path):
    with open(file_path, "r") as f:
        interarrival_times = json.load(f)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] Interarrival times loaded from {file_path}.")
    return interarrival_times

def poisson_arrival(lambd):
    interArrivalTime = random.expovariate(lambd)
    return interArrivalTime

def simulate_requests(num_requests, lambd, ip, model):
    interarrival_times = []
    for req in range(num_requests):
        interarrival_time = poisson_arrival(lambd)
        interarrival_times.append(interarrival_time)
        #threading.Thread(target=request, args=(req, ip, model)).start()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] mean IAT - {np.mean(interarrival_times)}")
    return interarrival_times

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="127.0.0.1", help="ip address")
    parser.add_argument("--num_requests", "-n", type=int, help="num of requests")
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2", help="model name")
    parser.add_argument("--poisson_lambda", "-l", default=37, type=float, help="poisson arrival rate")
    parser.add_argument("--save_IAT", "-s", type=str, help="save arrival rate to json")
    parser.add_argument("--load_IAT", "-f", type=str, help="load interarrival times from json")
    parser.add_argument("--port", "-p", type=str, default="8000",help="port for api server. default=8000")
    #add dry run option store true
    parser.add_argument("--dry_run", "-d", action="store_true",help="only save IAT file. not querying")
    args = parser.parse_args()

    global prompt
    texts = f"""In the vast expanse of the universe, where countless galaxies dance in a cosmic symphony, the wonders of existence unfold before our eyes. From the smallest particles that weave the tapestry of reality to the majestic celestial bodies that adorn the night sky, every corner of this magnificent cosmos holds secrets waiting to be discovered. Through the lens of science, humanity has embarked on an awe-inspiring journey of exploration and understanding, unraveling the mysteries of life, the universe, and everything in between. Throughout history, we have witnessed remarkable leaps in knowledge and technology, propelling us forward in our quest for discovery. From the ancient philosophers who pondered the nature of existence to the Renaissance thinkers who ushered in an era of scientific enlightenment, each generation has built upon the knowledge of its predecessors, pushing the boundaries of human understanding ever further. They laid the foundation for the scientific revolution, which sparked an insatiable curiosity to explore the unknown. In the 21st century, humanity finds itself at the precipice of unprecedented advancements. Breakthroughs in fields such as artificial intelligence, biotechnology, and space exploration have the potential to reshape our world in profound ways. We stand on the cusp of unlocking the mysteries of the human brain, decoding the building blocks of life itself, and venturing beyond our planetary borders. With each passing day, scientists and innovators around the globe strive to unlock the secrets of our universe, harnessing the power of technology to tackle the grand challenges that lie ahead. Space agencies and private companies collaborate to send missions to distant planets, seeking clues about our origins and the possibility of extraterrestrial life. We send probes and telescopes deeper into space, capturing breathtaking images and collecting data that expands our understanding of the cosmos. Yet, amidst the rapid progress and scientific marvels, we must not forget the importance of preserving our fragile planet and fostering a harmonious coexistence with the natural world. Climate change, habitat destruction, and pollution threaten the very ecosystems that sustain life on Earth. It is our collective responsibility to safeguard this precious home and seek sustainable solutions for a better future. Moreover, as we navigate the complexities of a globalized world, it is crucial to embrace diversity, empathy, and inclusivity. Our differences in culture, religion, and perspective should be celebrated as strengths that enrich our collective human experience.  By fostering collaboration and understanding, we can overcome the challenges that divide us, forging a path towards a more compassionate and united global society. In the realms of art, literature, and music, human creativity continues to flourish, serving as a testament to our capacity for imagination and expression. Through masterful strokes of a paintbrush, captivating words on a page, or melodies that resonate with our souls, artists shape our understanding of the world, evoke emotions, and challenge our perceptions. They give voice to the human experience and inspire us to see the beauty in everyday life. Art reminds us of our shared humanity and the power of beauty to transcend the boundaries of language and culture. It connects us at a deeper level, allowing us to explore the depths of our emotions and ignite a sense of wonder within us. In galleries, theaters, and concert halls, we are transported to different worlds and given glimpses into the minds of those who create. In the digital age, the rapid expansion of information and communication technologies has transformed the way we connect, learn, and interact. Social media platforms, virtual reality, and the internet have opened up new avenues for global dialogue and collaboration. The world has become more interconnected than ever before, bridging distances and enabling people from different corners of the globe to exchange ideas and cultures instantaneously. As we gaze into the future, the possibilities that lie ahead are both exhilarating and humbling. From the exploration of distant planets to the potential for groundbreaking scientific discoveries, humanity's journey of discovery and self-discovery continues to unfold. It is a reminder that we are part of something greater, a cosmic tapestry of existence, bound together by the shared wonder and curiosity that defines us as human beings. So, let us embark on this grand adventure together,"""

    prompt = f"Translate the ```{texts}``` to French"
    openai.api_key = "EMPTY" # Not support yet
    
    #print(os.path.exists(os.path.join(os.getcwd(), args.load_IAT)))
    if args.load_IAT and os.path.exists(os.path.abspath(args.load_IAT)):
        interarrival_times = load_interarrival_times(args.load_IAT)
    else:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] IAT file not found or not specified. Generating new interarrival times.")
        interarrival_times = simulate_requests(args.num_requests, args.poisson_lambda , args.ip, args.model)

    # Save interarrival times to JSON file if not loaded from file
    if not args.load_IAT and args.save_IAT:
        with open(f"{args.save_IAT}_{args.num_requests}reqs_lambd{args.poisson_lambda}.json", "w") as f:
            json.dump(interarrival_times, f)
        print(f"IAT {args.save_IAT}_{args.num_requests}reqs_lambd{args.poisson_lambda}.json saved !")
        if args.dry_run:
            print(f"exit since dry run specified")
            exit(0)

    t_start = time.time()
    req_ID = 1
    arrival_times, curr_time = [], 0
    for interarrival_time in interarrival_times:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] interArrivalTime: {interarrival_time}")
        time.sleep(interarrival_time)
        curr_time += interarrival_time
        arrival_times.append(curr_time)
        
        threading.Thread(target=request, args=(req_ID, args.ip, args.port, args.model)).start()
        req_ID += 1

    t_end = time.time() - t_start
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] {args.num_requests} requests ends with {t_end-t_start}")



if __name__ == "__main__":
    main()
