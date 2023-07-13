# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-10b", trust_remote_code=True)
# model = AutoModelForSeq2SeqLM.from_pretrained("THUDM/glm-10b", trust_remote_code=True)
# model = model.half().cuda()
# model.eval()

# # Inference
# inputs = tokenizer("Ng is an adjunct professor at [MASK] (formerly associate professor and Director of its Stanford AI Lab or SAIL ). Also a pioneer in online education, Ng co-founded Coursera and deeplearning.ai.", return_tensors="pt")
# inputs = tokenizer.build_inputs_for_generation(inputs, max_gen_length=512)
# inputs = inputs.to('cuda')
# outputs = model.generate(**inputs, max_length=512, eos_token_id=tokenizer.eop_token_id)
# print(tokenizer.decode(outputs[0].tolist()))

# # Training
# inputs = tokenizer(
#     ["Tsinghua University is located in [MASK].", "One minus one equals zero, is it correct? Answer: [MASK]"],
#     return_tensors="pt", padding=True)
# inputs = tokenizer.build_inputs_for_generation(inputs, targets=["Beijing", "No"], max_gen_length=8, padding=False)
# inputs = inputs.to('cuda')
# outputs = model(**inputs)
# loss = outputs.loss
# logits = outputs.logits
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-10b-chinese", trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained("THUDM/glm-10b-chinese", trust_remote_code=True)
model = model.half().cuda()

inputs = tokenizer("凯旋门位于意大利米兰市古城堡旁。1807年为纪念[MASK]而建，门高25米，顶上矗立两武士青铜古兵车铸像。", return_tensors="pt")
inputs = tokenizer.build_inputs_for_generation(inputs, max_gen_length=512)
inputs = {key: value.cuda() for key, value in inputs.items()}
outputs = model.generate(**inputs, max_length=512, eos_token_id=tokenizer.eop_token_id)
print(tokenizer.decode(outputs[0].tolist()))