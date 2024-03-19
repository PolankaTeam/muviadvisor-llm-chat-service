import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH: str = 'models/phi-2'

torch.set_default_device("cuda")

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH,
                                            torch_dtype='auto',
                                            trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

inputs = tokenizer('''def print_prime(n):
   """
   Print all primes between 1 and n
   """''', return_tensors="pt", return_attention_mask=False)

outputs = model.generate(**inputs, max_length=200)
text = tokenizer.batch_decode(outputs)[0]
print(text)