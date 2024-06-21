import re
from transformers import (
    BatchEncoding,
)
from transformers import AutoTokenizer


class TokenizerAligner:
    @staticmethod
    def contains_emoji_or_greek(s):
        emoji_and_greek_pattern = re.compile(
            "["
            "\U0001f600-\U0001f64f"  # Emoticons
            "\U0001f300-\U0001f5ff"  # Miscellaneous Symbols and Pictographs
            "\U0001f680-\U0001f6ff"  # Transport and Map Symbols
            "\U0001f1e0-\U0001f1ff"  # Regional Indicator Symbols
            "\U00002600-\U000026ff"  # Miscellaneous Symbols
            "\U00002700-\U000027bf"  # Dingbats
            "\U0001f900-\U0001f9ff"  # Supplemental Symbols and Pictographs
            "\U0001f018-\U0001f270"  # Various Asian characters
            "\U0001f201-\U0001f251"  # Enclosed Ideographic Supplement
            "\U0001f004-\U0001f0f5"  # Mahjong Tiles
            "\U0001f0a0-\U0001f0ae"  # Playing Cards
            "\U00000370-\U000003ff"  # Greek and Coptic
            "\U0001f700-\U0001f77f"  # Alchemical Symbols
            "]+",
            flags=re.UNICODE,
        )
        return bool(emoji_and_greek_pattern.search(s))

    @staticmethod
    def text_to_words(text, text_tokenized: BatchEncoding, word_token_ids: list):
        words = []
        for i in range(1 + max([x if x is not None else 0 for x in word_token_ids])):
            start, end = text_tokenized.word_to_chars(i)
            words.append(text[start:end].strip().lower())
        return words

    @staticmethod
    def text_to_words_batch(texts: list, text_tokenized_all: BatchEncoding):
        words_all = []
        for i in range(len(text_tokenized_all["input_ids"])):
            words = []
            for j in range(
                1
                + max(
                    [x if x is not None else 0 for x in text_tokenized_all[i].word_ids]
                )
            ):
                chars = text_tokenized_all[i].word_to_chars(j)
                words.append(texts[i][chars[0] : chars[1]].strip().lower())
            words_all.append(words)
        return words_all

    @staticmethod
    def _map_words_v2(main_list: list, compare_list: list, i: int, j: int):
        result_ids, result_words = [], []
        special_characters = ["\u200b", "\u200d"]
        k, l = 1, 1
        while (i + l) <= len(main_list) and (j + k) <= len(compare_list):
            if "".join(main_list[i : i + l]) == "".join(compare_list[j : j + k]):
                result_ids.append((list(range(i, i + l)), list(range(j, j + k))))
                result_words.append(
                    (
                        "".join(main_list[i : i + l]),
                        "".join(compare_list[j : j + k]),
                    )
                )
                if k > 10 and l > 10:
                    raise Exception(
                        "Error in the alignment of the tokens to much tokens"
                    )
                i += l
                j += k
                return result_ids, result_words, i, j
            elif "".join(main_list[i : i + l]) in "".join(compare_list[j : j + k]):
                l = l + 1
            elif "".join(compare_list[j : j + k]) in "".join(main_list[i : i + l]):
                k = k + 1
            elif "".join(compare_list[j : j + k]) == "".join(
                main_list[i + 1 : i + l + 1]
            ):
                result_ids.append((list(range(i, i + l + 1)), list(range(j, j + k))))
                result_words.append(
                    (
                        "".join(main_list[i : i + l + 1]),
                        "".join(compare_list[j : j + k]),
                    )
                )
                i += l + 1
                j += k
                return result_ids, result_words, i, j
            elif "".join(compare_list[j + 1 : j + k + 1]) == "".join(
                main_list[i : i + l]
            ):
                result_ids.append((list(range(i, i + l)), list(range(j, j + k + 1))))
                result_words.append(
                    (
                        "".join(main_list[i : i + l]),
                        "".join(compare_list[j : j + k + 1]),
                    )
                )
                i += l
                j += k + 1
                return result_ids, result_words, i, j
            else:
                # some examples
                if "".join(compare_list[j : j + k]) in "".join(
                    main_list[i : i + l + 1]
                ):
                    l = l + 1
                elif "".join(main_list[i : i + l]) in "".join(
                    compare_list[j : j + k + 1]
                ):
                    k = k + 1
                elif "".join(compare_list[j : j + k + 1]) in "".join(
                    main_list[i : i + l]
                ):
                    k = k + 1
                elif "".join(compare_list[j : j + k]) in "".join(
                    main_list[i : i + l + 1]
                ):
                    l = l + 1
                else:
                    emoji_main = [
                        TokenizerAligner.contains_emoji_or_greek(s)
                        for s in main_list[i : i + l]
                    ]
                    emoji_compare = [
                        TokenizerAligner.contains_emoji_or_greek(s)
                        for s in compare_list[j : j + k]
                    ]
                    if True in emoji_main or True in emoji_compare:
                        while TokenizerAligner.contains_emoji_or_greek(
                            main_list[i + l]
                        ):
                            l = l + 1
                        while TokenizerAligner.contains_emoji_or_greek(
                            compare_list[j + k]
                        ):
                            k = k + 1
                        result_ids.append(
                            (list(range(i, i + l)), list(range(j, j + k)))
                        )
                        result_words.append(
                            (
                                "".join(main_list[i : i + l]),
                                "".join(compare_list[j : j + k]),
                            )
                        )
                        i += l
                        j += k
                        return result_ids, result_words, i, j
                    print("problema")
                    print("".join(main_list[i]), "".join(compare_list[j]))
                    raise Exception(f"Error in the alignment of the tokens i{i} j{j}")
                i = i + 1
                j = j + 1
        result_ids.append(
            (list(range(i, len(main_list) + 1)), list(range(j, len(compare_list) + 1)))
        )
        result_words.append(
            (
                "".join(main_list[i:]),
                "".join(compare_list[j:]),
            )
        )
        i = len(main_list)
        j = len(compare_list)
        return result_ids, result_words, i, j

    @staticmethod
    def _map_words(
        i, j, comb_sorted, main_list, compare_list, result_ids, result_words
    ):
        asigned = 0
        for k, l in comb_sorted:
            if "".join(main_list[i : i + l]) == "".join(compare_list[j : j + k]):
                result_ids.append((list(range(i, i + l)), list(range(j, j + k))))
                result_words.append(
                    (
                        "".join(main_list[i : i + l]),
                        "".join(compare_list[j : j + k]),
                    )
                )
                i += l
                j += k
                asigned = 1
                break
        return asigned, i, j, result_ids, result_words

    @staticmethod
    def map_words(main_list: list, compare_list: list, n: int = 15):
        result_ids, result_words = [], []
        i = 0
        j = 0
        comb = [(i, j) for j in range(1, n) for i in range(1, n)]
        main_list = [re.sub(r"\s+", "", word) for word in main_list]
        compare_list = [re.sub(r"\s+", "", word) for word in compare_list]
        # we sorted the combinations to prioritize the most probables
        comb_sorted = sorted(comb, key=lambda x: x[0] + x[1], reverse=False)
        while i < len(main_list) and j < len(compare_list):
            asigned, i, j, result_ids, result_words = TokenizerAligner._map_words(
                i, j, comb_sorted, main_list, compare_list, result_ids, result_words
            )
            if asigned == 1:
                continue
            else:
                result_ids_plus, result_words_plus, i, j = (
                    TokenizerAligner._map_words_v2(main_list, compare_list, i, j)
                )
                result_ids.extend(result_ids_plus)
                result_words.extend(result_words_plus)
        return result_ids, result_words

    @staticmethod
    def map_tokens(
        words_ids_mapped: list, word_ids_first_list: list, word_ids_second_list: list
    ):
        tokens_ids_mapped = []
        for words in words_ids_mapped:
            tokens_id_first, tokens_id_second = [], []
            for word_id_first in words[0]:
                tokens_id_first.extend(
                    list(
                        index
                        for index, value in enumerate(word_ids_first_list)
                        if value == word_id_first
                    )
                )
            for word_id_second in words[1]:
                tokens_id_second.extend(
                    list(
                        index
                        for index, value in enumerate(word_ids_second_list)
                        if value == word_id_second
                    )
                )
            tokens_ids_mapped.append((tokens_id_first, tokens_id_second))
        return tokens_ids_mapped

    @staticmethod
    def _map_tokens_to_str_idx(
        tokens_idx_mapped: list, tokens_str_first: list, tokens_str_second: list
    ):
        tokens_str_mapped = []
        for tokens in tokens_idx_mapped:
            tokens_str_mapped.append(
                (
                    [tokens_str_first[idx] for idx in tokens[0]],
                    [tokens_str_second[idx] for idx in tokens[1]],
                )
            )
        return tokens_str_mapped

    @staticmethod
    def _map_tokens_to_str_batch(
        tokens_idx_mapped: list,
        text_tokenized_first: BatchEncoding,
        text_tokenized_second: BatchEncoding,
    ):
        token_str_mapped = []
        for i in range(len(text_tokenized_first["input_ids"])):
            tokens_str_first = text_tokenized_first[i].tokens
            tokens_str_second = text_tokenized_second[i].tokens

            token_str_mapped_i = TokenizerAligner._map_tokens_to_str_idx(
                tokens_idx_mapped[i], tokens_str_first, tokens_str_second
            )
            token_str_mapped.append(token_str_mapped_i)
        return token_str_mapped

    @staticmethod
    def _map_tokens_to_str(
        tokens_idx_mapped: list,
        text_tokenized_first: BatchEncoding,
        text_tokenized_second: BatchEncoding,
    ):
        tokens_str_first = text_tokenized_first.tokens()
        tokens_str_second = text_tokenized_second.tokens()

        tokens_str_mapped = TokenizerAligner._map_tokens_to_str_idx(
            tokens_idx_mapped, tokens_str_first, tokens_str_second
        )
        return tokens_str_mapped

    @staticmethod
    def map_tokens_to_str(
        tokens_idx_mapped: list,
        text_tokenized_first: BatchEncoding,
        text_tokenized_second: BatchEncoding,
    ):
        if isinstance(text_tokenized_first["input_ids"][0], list):
            tokens_str_mapped = TokenizerAligner._map_tokens_to_str_batch(
                tokens_idx_mapped=tokens_idx_mapped,
                text_tokenized_first=text_tokenized_first,
                text_tokenized_second=text_tokenized_second,
            )
        else:
            tokens_str_mapped = TokenizerAligner._map_tokens_to_str(
                tokens_idx_mapped=tokens_idx_mapped,
                text_tokenized_first=text_tokenized_first,
                text_tokenized_second=text_tokenized_second,
            )

        return tokens_str_mapped

    @staticmethod
    def _map_tokens_to_id_idx(
        tokens_idx_mapped: list, tokens_id_first: list, tokens_id_second: list
    ):
        tokens_str_mapped = []
        for tokens in tokens_idx_mapped:
            tokens_str_mapped.append(
                (
                    [tokens_id_first[idx] for idx in tokens[0]],
                    [tokens_id_second[idx] for idx in tokens[1]],
                )
            )
        return tokens_str_mapped

    @staticmethod
    def _map_tokens_to_id(
        tokens_idx_mapped: list,
        text_tokenized_first: BatchEncoding,
        text_tokenized_second: BatchEncoding,
    ):
        tokens_id_mapped = []

        tokens_id_first = text_tokenized_first["input_ids"]
        tokens_id_second = text_tokenized_second["input_ids"]

        tokens_id_mapped = TokenizerAligner._map_tokens_to_id_idx(
            tokens_idx_mapped, tokens_id_first, tokens_id_second
        )
        return tokens_id_mapped

    @staticmethod
    def _map_tokens_to_id_batch(
        tokens_idx_mapped: list,
        text_tokenized_first: BatchEncoding,
        text_tokenized_second: BatchEncoding,
    ):
        tokens_id_mapped = []
        for i in range(len(text_tokenized_first["input_ids"])):
            tokens_id_first = text_tokenized_first["input_ids"][i]
            tokens_id_second = text_tokenized_second["input_ids"][i]

            tokens_id_mapped_i = TokenizerAligner._map_tokens_to_id_idx(
                tokens_idx_mapped[i], tokens_id_first, tokens_id_second
            )
            tokens_id_mapped.append(tokens_id_mapped_i)
        return tokens_id_mapped

    @staticmethod
    def map_tokens_to_id(
        tokens_idx_mapped: list,
        text_tokenized_first: BatchEncoding,
        text_tokenized_second: BatchEncoding,
    ):
        if isinstance(text_tokenized_first["input_ids"][0], list):
            tokens_id_mapped = TokenizerAligner._map_tokens_to_id_batch(
                tokens_idx_mapped=tokens_idx_mapped,
                text_tokenized_first=text_tokenized_first,
                text_tokenized_second=text_tokenized_second,
            )
        else:
            tokens_id_mapped = TokenizerAligner._map_tokens_to_id(
                tokens_idx_mapped=tokens_idx_mapped,
                text_tokenized_first=text_tokenized_first,
                text_tokenized_second=text_tokenized_second,
            )

        return tokens_id_mapped

    @staticmethod
    def _words_tokens_to_str(word_token_ids: list, tokens_str: list):
        token_idx_words = []
        for word_id in range(
            1 + max([x if x is not None else 0 for x in word_token_ids])
        ):
            token_idx_words.append(
                list(
                    index
                    for index, value in enumerate(word_token_ids)
                    if value == word_id
                )
            )

        tokens_str_words = []
        for tokens in token_idx_words:
            tokens_str_words.append(([tokens_str[idx] for idx in tokens],))
        return tokens_str_words

    @staticmethod
    def words_tokens_to_str(
        word_token_ids: list,
        text_tokenized: BatchEncoding,
    ):
        tokens_str = text_tokenized.tokens()
        token_str_mapped = TokenizerAligner._words_tokens_to_str(
            word_token_ids, tokens_str
        )

        return token_str_mapped

    @staticmethod
    def _align_tokens(
        text: str,
        text_tokenized_first: BatchEncoding,
        text_tokenized_second: BatchEncoding,
    ):
        word_token_idx_first = text_tokenized_first.word_ids()
        word_token_idx_second = text_tokenized_second.word_ids()

        words_first = TokenizerAligner.text_to_words(
            text, text_tokenized_first, word_token_idx_first
        )
        words_second = TokenizerAligner.text_to_words(
            text, text_tokenized_second, word_token_idx_second
        )
        words_idx_mapped, words_str_mapped = TokenizerAligner.map_words(
            words_first, words_second
        )
        tokens_idx_mapped = TokenizerAligner.map_tokens(
            words_idx_mapped, word_token_idx_first, word_token_idx_second
        )
        return tokens_idx_mapped, words_str_mapped

    @staticmethod
    def _align_tokens_batch(
        text: str, text_tokenized_first: list, text_tokenized_second: list
    ):
        word_token_idx_first = [
            text_tokenized_first[i].word_ids
            for i in range(len(text_tokenized_first["input_ids"]))
        ]
        word_token_idx_second = [
            text_tokenized_second[i].word_ids
            for i in range(len(text_tokenized_second["input_ids"]))
        ]
        words_first = TokenizerAligner.text_to_words_batch(text, text_tokenized_first)
        words_second = TokenizerAligner.text_to_words_batch(text, text_tokenized_second)
        tokens_idx_mapped, words_str_mapped = [], []
        for i in range(len(text_tokenized_first["input_ids"])):
            words_idx_mapped_i, words_str_mapped_i = TokenizerAligner.map_words(
                words_first[i], words_second[i]
            )
            tokens_idx_mapped_i = TokenizerAligner.map_tokens(
                words_idx_mapped_i, word_token_idx_first[i], word_token_idx_second[i]
            )
            tokens_idx_mapped.append(tokens_idx_mapped_i)
            words_str_mapped.append(words_str_mapped_i)

        return tokens_idx_mapped, words_str_mapped

    @staticmethod
    def align_tokens(
        text: str,
        text_tokenized_first: list,
        text_tokenized_second: list,
        return_all=False,
    ):
        """
        Aligns the tokens of two tokenized texts using the word of the tokenized texts.
        Fist aligns the words of the texts and then maps the tokens of the tokenized texts to the words.
        Args:
            text: The original text.
            text_tokenized_first: The first tokenized text.
            text_tokenized_second: The second tokenized text.
            return_words: If True, returns the words of the texts.
        Returns:
            tokens_idx_mapped: A list of tuples with the token indices of the first and second tokenized texts mapped. (one row per words mapped.)
            tokens_id_mapped: A list of tuples with the token ids of the first and second tokenized texts mapped. (one row per words mapped.)
            words_str_mapped: A list of tuples with the words str of the first and second texts mapped.
        """
        if isinstance(text_tokenized_first["input_ids"][0], list):
            tokens_idx_mapped, words_str_mapped = TokenizerAligner._align_tokens_batch(
                text, text_tokenized_first, text_tokenized_second
            )
            tokens_id_mapped = TokenizerAligner._map_tokens_to_id_batch(
                tokens_idx_mapped, text_tokenized_first, text_tokenized_second
            )
        else:
            tokens_idx_mapped, words_str_mapped = TokenizerAligner._align_tokens(
                text, text_tokenized_first, text_tokenized_second
            )
            tokens_id_mapped = TokenizerAligner._map_tokens_to_id(
                tokens_idx_mapped, text_tokenized_first, text_tokenized_second
            )

        if return_all:
            tokens_str_mapped = TokenizerAligner.map_tokens_to_str(
                tokens_idx_mapped, text_tokenized_first, text_tokenized_second
            )
            return (
                tokens_idx_mapped,
                tokens_id_mapped,
                words_str_mapped,
                tokens_str_mapped,
            )

        return tokens_id_mapped
