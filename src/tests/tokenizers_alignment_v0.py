import os
import sys
import threading
from datetime import datetime


# sys.path.append('/home/csp/repo/LLMs/eye_transformer/')
# print("CWD", os.getcwd(), "PATH", sys.path)

import pathlib

sys.path.append("../..")

path = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(path)
path = str(pathlib.Path(__file__).parent.resolve().parent.resolve())
sys.path.append(path)
path = str(pathlib.Path(__file__).parent.resolve().parent.resolve().parent.resolve())
sys.path.append(path)
path = str(
    pathlib.Path(__file__)
    .parent.resolve()
    .parent.resolve()
    .parent.resolve()
    .parent.resolve()
)
sys.path.append(path)
from transformers import AutoTokenizer
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union
from transformers import (
    LlamaForSequenceClassification,
    AutoModelForSequenceClassification,
)
from transformers.modeling_outputs import (
    SequenceClassifierOutputWithPast,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from fixations_predictor.utilsFP import FixationsPredictor
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
from typing import (
    TypeVar,
)

from datasets import load_dataset

import pandas as pd
import matplotlib.pyplot as plt
from operator import itemgetter
from functools import partial
from utils.dataset_proceser import DatasetProceser

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
B_TEXT, B_TEXT = "<s>", "</s>"

from typing import List, Tuple
import re


# THIS IS A FIRST VERSION THAT NEVER WORKED
def match_tokens_to_words_llama(tokenizer, text=None, tokens_id=None):
    if tokens_id is None:
        tokens_id = tokenizer.tokenize(text)
    if text is None:
        text = tokenizer.decode(tokens_id, skip_special_tokens=True)
    if tokens_id is None and text is None:
        raise ValueError("Either text or tokens must be provided")

    tokens = tokenizer.tokenize(text)
    tokens_id = tokenizer.convert_tokens_to_ids(tokens)
    # remove Ġ in each token
    # tokens = [token.replace("Ġ", "") for token in tokens]
    words = re.findall(r"\S+|\n", text)
    token_index = 0
    word_to_tokens = []

    for word in words:
        word_tokens_id = []
        accumulated_token = ""
        word_pointer = 0
        current_tokens = []
        valid = True

        while token_index < len(tokens) and word_pointer < len(word):
            token = tokens[token_index]
            token_cleaned = token[1:] if token.startswith("Ġ") else token

            accumulated_token += token_cleaned
            current_tokens.append(tokens_id[token_index])
            token_index += 1

            if accumulated_token == word[: len(accumulated_token)]:
                word_pointer += len(token_cleaned)
                if word_pointer == len(word):
                    word_tokens_id.extend(current_tokens)
                    current_tokens = []
                    break
            elif not word.startswith(accumulated_token):
                valid = False
                break

        if valid:
            word_to_tokens.append((word, word_tokens_id))
        else:
            # Push back the token index and reset current tokens if matching failed
            token_index -= len(current_tokens)
            current_tokens = []
            word_to_tokens.append((word, []))

    return word_to_tokens, text, tokens


def match_tokens_to_words_t5(tokenizer, text=None, tokens=None):
    if tokens is None:
        tokens = tokenizer.tokenize(text)
    if text is None:
        text = tokenizer.decode(tokens, skip_special_tokens=True)
    if tokens is None and text is None:
        raise ValueError("Either text or tokens must be provided")

    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    word_to_tokens = {}
    token_index = 0

    # Helper function to strip special characters used in tokenization
    def clean_token(token):
        return token.replace(
            "▁", ""
        )  # T5 uses the '▁' character to indicate the start of a new word

    words = text.split()
    for word in words:
        current_word = word
        current_token_ids = []
        while token_index < len(tokens) and current_word:
            token = tokens[token_index]
            clean_token_text = clean_token(token)
            if current_word.startswith(clean_token_text):
                token_length = len(clean_token_text)
                current_token_ids.append(token_ids[token_index])
                current_word = current_word[token_length:]
                token_index += 1
            else:
                break
        word_to_tokens[word] = current_token_ids

    return word_to_tokens, text, tokens


tokenizer_name = "t5-small"
tokenizer_t5 = AutoTokenizer.from_pretrained(
    tokenizer_name, cache_dir="./cache/models", model_max_length=2048
)

# tokenizer = T5Tokenizer.from_pretrained('t5-small')
tokenizer_name = "meta-llama/Meta-Llama-3-8B"
tokenizer_llama = AutoTokenizer.from_pretrained(
    tokenizer_name, cache_dir="./cache/models", model_max_length=2048
)
tokenizer_llama.add_special_tokens({"pad_token": "[PAD]"})
dataset = "timdettmers/openassistant-guanaco"
data = load_dataset(dataset, split="train[:2%]")


def tokenize_function(example, tokenizer):
    return tokenizer(example["text"], padding=True, truncation=True)


# Apply the mapping function

function_process = partial(tokenize_function, tokenizer=tokenizer_llama)
data = data.map(
    function_process,
    batched=True,
)

batch_size = 2
data.set_format(columns=["input_ids", "attention_mask"])
# tokens = [tokenizer(t['text']) for t in dataset]
collator = DataCollatorWithPadding(tokenizer_llama)
dataloader = torch.utils.data.DataLoader(
    dataset=data, batch_size=batch_size, collate_fn=collator
)

for batch in dataloader:
    print(batch)
    for input_ids in batch["input_ids"]:
        tokens_to_words_llama, text, tokens = match_tokens_to_words_llama(
            tokenizer_llama, tokens_id=input_ids
        )
        print(tokens_to_words_llama)
        for tok in tokens_to_words_llama:
            print(tok)
        tokens_to_words_t5, text, tokens = match_tokens_to_words_t5(
            tokenizer_t5, text=text
        )
        print(tokens_to_words_t5)
    break
