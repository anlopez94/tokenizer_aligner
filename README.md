# tokenizer_aligner

# tokenizer_aligner

tokenizer_aligner is a Python library for mapping tokens between different tokenizers

## Modules

#### fixations_predictor:
First, we perform an initial mapping of tokens to the words they belong to in each tokenizer with some properties of \textit{FastTokenizers} from the \textit{transformers} library. Then, we map words from one tokenizer to the words in the other and finally, we assume that the combination of the tokens that are mapped to a word in one tokenizer correspond to the tokens that are mapped to the word that is mapped to the initial word in the other tokenizer. 

**Token Mapping Example**


|  | Tokenizer 1 | Tokenizer 2 |
|-------|-------------|-------------|
| Words | astrophotography | astrophotography |
| Tokens | ['_Astro', 'photo', 'graphy'] | ['Ċ', 'Ast', 'roph', 'ot', 'ography'] |
| Token indices | [22, 23, 24] | [23, 24, 25, 26, 271] |
| Token IDs | [15001, 17720, 16369] | [198, 62152, 22761, 354, 5814] |


## Installation

```sh
pip install git+https://github.com/anlopez94/tokenizer_aligner.git
```





