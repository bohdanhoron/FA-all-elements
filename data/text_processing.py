from string import punctuation
from typing import List, Union

from data.tokenizer import CodeTokenizer
from utils.decorators import memoize
import re

def remove_punctuation_for_words(data: str) -> List[str]:
    """
    Видаляє пунктуацію з тексту та розбиває на слова.
    
    Args:
        data: Вхідний текст
        
    Returns:
        List[str]: Список слів без пунктуації
    """
    translator = str.maketrans('', '', punctuation)
    cleaned = data.translate(translator)
    return cleaned.split()


class NgrammProcessor:
    def __init__(self, ignore_punctuation: bool = True, computer_code: bool = False, ignore_comments: bool = False):
        self.ignore_punctuation = ignore_punctuation
        self.computer_code = computer_code
        self.ignore_comments = ignore_comments
        self.words = []
        self.processed_text = ""

    def preprocess(self, text: str, file_name: str = None) -> None:
        if self.computer_code:
            code_tokenizer = CodeTokenizer(text, file_name)
            result = code_tokenizer.process()

            if not self.ignore_comments:
                for token in result[:]:
                    if (("Comment" in token["type"] or "Doc" in token["type"]) and token["type"] != "Comment.Special"):
                        comment_words = self.__process_words(token["value"], False)
                        if comment_words and len(comment_words) > 0:
                            result.remove(token)
                            self.words += comment_words
            else:
                result = [item for item in result if "Comment" not in item["type"] and "Doc" not in item["type"]]

            self.words += [item["value"] for item in result]
            return

        self.words = self.__process_words(text)

    def __process_words(self, text: str, lower_case: bool = True) -> List[str]:
        if self.ignore_punctuation:
            self.processed_text = ''.join(char for char in text if char not in punctuation or char == '-' or char == "'")
        else:
            self.processed_text = text

        if lower_case:
            return [word.lower() for word in re.findall(r'\b\w+(?:[-\']\w+)*\b', self.processed_text)]
        else:
            return [word for word in re.findall(r'\b\w+(?:[-\']\w+)*\b', self.processed_text)]

    def get_words(self, remove_empty_entries: bool = False) -> List[str]:
        if remove_empty_entries:
            return [word for word in self.words if word]
        return self.words


def remove_punctuation(data):
    temp = []
    for i in range(len(data)):
        if data[i] in punctuation:
            continue
        else:
            temp.append(data[i].lower())
    return "".join(temp)

def is_valid_letter(char: str) -> bool:
    """Check if a character should be skipped."""
    invalid_characters = [' ', '\n', '\ufeff', '°', '"', '„', '–']
    return char in invalid_characters


@memoize
def prepare_data(data: str, n: int, split: str, filename: str, computer_code: bool = False, ignore_comments: bool = False) -> List:
    """Підготовка даних для аналізу."""
    if n is None:
        return no_update
    
    if not computer_code:
        data = re.sub(r'\n+', '\n', data)
        data = re.sub(r'\n\s\s', '\n', data)
        data = re.sub(r'﻿', '', data)
    
    if n == 1:
        if split == "word":
            if not computer_code:
                data = re.sub(r'--', ' -', data)
            processor = NgrammProcessor(computer_code=computer_code, ignore_comments=ignore_comments)
            processor.preprocess(data, file_name=filename)
            result = processor.get_words()
            return result
            
        elif split == 'letter':
            temp = []
            data = remove_punctuation(data)
            for word in data:
                for i in word:
                    if i == ' ':
                        continue
                    temp.append(i)
            return temp
            
        elif split == 'symbol':
            result = []
            for char in data:
                if char == " " or char == "\n" or char == "\ufeff":
                    result.append("space")
                else:
                    result.append(char.lower())
            return result
    else:
        if split == "word":
            data = re.sub(r'--', ' -', data)
            processor = NgrammProcessor()
            processor.preprocess(data)
            words = processor.get_words()
            L = len(words)
            
            result = []
            for i in range(L - n + 1):
                window = tuple(words[i:i + n])
                result.append(window)
            return result
                
        elif split == "letter":
            temp = []
            data = remove_punctuation(data)
            for word in data:
                for i in word:
                    if i == ' ':
                        continue
                    temp.append(i)
            L = len(temp)
            data = temp
            temp = []
            for i in range(L - n + 1):
                window = tuple(data[i:i + n])
                temp.append(window)
            return temp
                
        elif split == 'symbol':
            temp = []
            for char in data:
                if char == " " or char == "\n" or char == "\ufeff":
                    temp.append("space")
                else:
                    temp.append(char.lower())
            data = temp
            L = len(data)
            temp = []
            for i in range(L - n + 1):
                window = tuple(data[i:i + n])
                temp.append(window)
            return temp

    return []
