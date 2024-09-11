from transformers import AutoTokenizer
import transformers
transformers.tokenization_utils_base.BatchEncoding

text = 'this is an example of a string'
def compute_words(tokenizer, text):
    
    text_tokenized = tokenizer(
                        text,
                        padding=False,
                        add_special_tokens=True,
                        return_tensors="pt",
                    )

    num_words = max([x if x is not None else 0 for x in text_tokenized.word_ids()])
    words = []
    for j in range(num_words):
        print(type(text_tokenized))
        chars = text_tokenized.word_to_chars(j)
        print(chars)
        chars_next = text_tokenized.word_to_chars(j+1)
        words.append(text[chars[0] : chars_next[0]].strip().lower())
    if num_words > 0:
        words.append(text[chars_next[0] : ].strip().lower())
    else:
        words.append(text.strip().lower())
    return words


print('----------------')
base_model = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(
                base_model, trust_remote_code=True
            )
print(base_model)
print(type(tokenizer))
words = compute_words(tokenizer, text)
print(words)
print('----------------')
base_model = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(
                base_model, trust_remote_code=True
            )
print(base_model)
print(type(tokenizer))
words = compute_words(tokenizer, text)
print(words)
print('----------------')
base_model = "mistralai/Mistral-7B-v0.3"
tokenizer = AutoTokenizer.from_pretrained(
                base_model, trust_remote_code=True
            )
print(base_model)
print(type(tokenizer))
words = compute_words(tokenizer, text)
print(words)
print('----------------')
base_model='roberta-base'
tokenizer = transformers.RobertaTokenizerFast.from_pretrained(base_model, add_prefix_space=True)
print(base_model)
print(type(tokenizer))
words = compute_words(tokenizer, text)
print(words)
print('----------------')
base_model="t5-small"
tokenizer = AutoTokenizer.from_pretrained(
            base_model, cache_dir="./cache/models", model_max_length=2048
        )
print(base_model)
print(type(tokenizer))
words = compute_words(tokenizer, text)
print(words)
print('----------------')