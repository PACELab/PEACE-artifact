from transformers import GPT2Tokenizer, OPTForCausalLM
"""
model = OPTForCausalLM.from_pretrained("facebook/opt-6.7b")
tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-6.7b")

prompt = "The benefits of deadlifting can be listed as below:"
inputs = tokenizer(prompt, return_tensors="pt").input_ids
"""

model = OPTForCausalLM.from_pretrained("facebook/opt-6.7b", device_map="auto")
tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-6.7b")

prompt = "Anti Vaccine Movemenet"
inputs = tokenizer(prompt, return_tensors="pt").input_ids.cuda()




"""
gen_tokens = model.generate(
    inputs,
    do_sample=True,
    temperature=0.9,
    max_length=1000,
)
gen_text = tokenizer.batch_decode(gen_tokens)[0]
"""
generate_ids = model.generate(inputs,max_length=2000,early_stopping= True,do_sample=True,min_length=2000,top_k=125,top_p=0.92,temperature= 0.85,repetition_penalty=1.5,num_return_sequences=3)

for i, sample_output in enumerate(generate_ids):
    result = tokenizer.decode(sample_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    print(result) 
    with open('d_result_'+str(i)+'.txt', 'w', encoding="utf-8") as f:
        f.writelines(result)    