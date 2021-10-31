# Azure Machine Learning Huggingface

## how to use huggingface model

- Run huggingface model in Azure Machine learning services
- Samples available from - https://huggingface.co/transformers/model_doc/prophetnet.html

## Prerequistie

- Azure Account
- Azure Storage
- Azure Machine learning Service
- Github account and repository
- Azure Service Principal Account
- Provide service principal contributor access to Machine learning resource
- Azure Keyvault to store secrets
- Update the keyvault with Service principal Secrets
- This automates the training code and Registers the model

## Steps

- Log into ml.azure.com
- Start your compute instance
- Go to notebook and start to write the code
- Note upgrade your aml sdk if needed

## Installations

```
pip install transformers
```

```
pip install transformers[torch]
```

## Code

- Test sentiment model

```
from transformers import pipeline; 
print(pipeline('sentiment-analysis')('we love you'))
```

- Let's try bert model

```
from transformers import BertTokenizer, FlaxBertForQuestionAnswering

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = FlaxBertForQuestionAnswering.from_pretrained('bert-base-uncased')

question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
inputs = tokenizer(question, text, return_tensors='jax')

outputs = model(**inputs)
start_scores = outputs.start_logits
end_scores = outputs.end_logits
```

- Now run GPT2 model

```
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")

prompt = "In a shocking finding, scientists discovered a herd of unicorns living in a remote, " \
         "previously unexplored valley, in the Andes Mountains. Even more surprising to the " \
         "researchers was the fact that the unicorns spoke perfect English."

input_ids = tokenizer(prompt, return_tensors="pt").input_ids

gen_tokens = model.generate(input_ids, do_sample=True, temperature=0.9, max_length=100,)
gen_text = tokenizer.batch_decode(gen_tokens)[0]
```

- Another GPT2

```
import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2DoubleHeadsModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = TFGPT2DoubleHeadsModel.from_pretrained('gpt2')

# Add a [CLS] to the vocabulary (we should train it also!)
num_added_tokens = tokenizer.add_special_tokens({'cls_token': '[CLS]'})

embedding_layer = model.resize_token_embeddings(len(tokenizer))  # Update the model embeddings with the new vocabulary size

choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]
encoded_choices = [tokenizer.encode(s) for s in choices]
cls_token_location = [tokens.index(tokenizer.cls_token_id) for tokens in encoded_choices]

input_ids = tf.constant(encoded_choices)[None, :]  # Batch size: 1, number of choices: 2
mc_token_ids = tf.constant([cls_token_location])  # Batch size: 1

outputs = model(input_ids, mc_token_ids=mc_token_ids)
lm_prediction_scores, mc_prediction_scores = outputs[:2]
```

- Prophnet sample

```
from transformers import ProphetNetTokenizer, ProphetNetEncoder
import torch

tokenizer = ProphetNetTokenizer.from_pretrained('microsoft/prophetnet-large-uncased')
model = ProphetNetEncoder.from_pretrained('patrickvonplaten/prophetnet-large-uncased-standalone')
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
```

- Open AI GPT 2

```
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer("Hello world")['input_ids']
tokenizer(" Hello world")['input_ids']
```

- Pegasus xsum

```
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
src_text = [
    """ PG&E stated it scheduled the blackouts in response to forecasts for high winds amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."""
]

model_name = 'google/pegasus-xsum'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
batch = tokenizer(src_text, truncation=True, padding='longest', return_tensors="pt").to(device)
translated = model.generate(**batch)
tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
assert tgt_text[0] == "California's largest electricity provider has turned off power to hundreds of thousands of customers."
```

- More to come later