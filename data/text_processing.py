import re
from string import punctuation
from typing import List, Tuple, Optional, Dict, Any, Union

from tokenizer import CodeTokenizer
from utils.decorators import memoize

# NOTE клас із С# для обробки слів
class NgrammProcessor:
    """
    Клас для обробки тексту і отримання n-грам.
    """
    def __init__(self, ignore_punctuation: bool = True, computer_code: bool = False, ignore_comments: bool = False):
        """
        Ініціалізує процесор n-грам.
        
        Args:
            ignore_punctuation: Чи ігнорувати пунктуацію при обробці
            computer_code:      Чи обробляти текст як комп'ютерний код
            ignore_comments:    Чи ігнорувати коментарі в коді
        """
        self.ignore_punctuation = ignore_punctuation
        self.computer_code = computer_code
        self.ignore_comments = ignore_comments
        self.words = []
        self.processed_text = ""
        
    def preprocess(self, text: str, file_name: str = None) -> None:
        """
        Попередня обробка тексту.
        
        Args:
            text: Вхідний текст для обробки
            file_name: назва файлу (потрібно для правильного визначення мови коду)
        """

        # Обробка тексту комп'ютерної програми
        if self.computer_code:
            code_tokenizer = CodeTokenizer(text, file_name)
            result = code_tokenizer.process()

            if not self.ignore_comments:
                # Передає коментарі на парсер натурального тексту
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
            
        # Розбиваємо текст на слова
        self.words = self.__process_words(text)
    
    def __process_words(self, text: str, lower_case: bool = True) -> List[str]:
        # Видаляємо пунктуацію, якщо потрібно
        if self.ignore_punctuation:
            # Використовуємо оптимізований метод видалення пунктуації
            self.processed_text = ''.join(char for char in text if char not in punctuation or char == '-' or char == "'")
        else:
            self.processed_text = text

        if lower_case:
            return [word.lower() for word in re.findall(r'\b\w+(?:[-\']\w+)*\b', self.processed_text)]
        else:
            return [word for word in re.findall(r'\b\w+(?:[-\']\w+)*\b', self.processed_text)]
        
    def get_words(self, remove_empty_entries: bool = False) -> List[str]:
        """
        Отримує список слів із обробленого тексту.
        
        Args:
            remove_empty_entries: Чи видаляти порожні рядки
            
        Returns:
            List[str]: Список слів
        """
        if remove_empty_entries:
            return [word for word in self.words if word]
        return self.words

def remove_empty_strings(arr: List[str]) -> List[str]:
    """
    Видаляє порожні рядки та спеціальні символи з списку.
    
    Args:
        arr: Список рядків для обробки
        
    Returns:
        List[str]: Список без порожніх рядків та спеціальних символів
    """
    return [item for item in arr if item and item != '\ufeff']


def is_valid_letter(char: str) -> bool:
    """
    Check if a character should be skipped.
    Returns True if character should be skipped.
    """
    invalid_characters = [' ', '\n', '\ufeff', '°', '"', '„', '–']
    return char in invalid_characters

def remove_punctuation_for_words(data):
    """
    Розбиває текст на слова та видаляє знаки пунктуації.
    
    Args:
        data: Вхідний текст
        
    Returns:
        List[str]: Список оброблених слів
    """
    # Використовуємо ефективніший регулярний вираз один раз
    words = re.findall(r'\b[a-zA-Z0-9]+(?:[-\'][a-zA-Z0-9]+)*\b', data.lower())
    
    # Обробляємо слова з дефісами та апострофами
    """result = []
    for word in words:
        if '-' in word or '\'' in word:
            # Розділяємо слово на підчастини за спеціальними символами
            parts = re.split(r'[-\']', word)
            # Додаємо лише непорожні частини
            result.extend([part for part in parts if part])
        else:
            result.append(word)"""
    
    #return result
    return words


def remove_punctuation(data):
    """
    Видаляє знаки пунктуації з тексту.
    """
    temp = []
    for i in range(len(data)):
        if data[i] in punctuation:
            continue
        else:
            temp.append(data[i].lower())
    return "".join(temp)


@memoize
def prepare_data(data: str, n: int, split: str, filename: str, computer_code: bool = False, ignore_comments: bool = False) -> List:
    """
    Підготовка даних для аналізу, розбиття на n-грами залежно від вказаних параметрів.
    
    Args:
        data: Вхідний текст для обробки
        n: Розмір n-грами
        split: Метод розбиття тексту ("word", "letter", "symbol")
        
    Returns:
        List: Список підготовлених даних
    """
    global L
    if n is None:
        return dash.no_update
    
    # Використовуємо спільний код попередньої обробки для всіх типів
    if not computer_code:
        data = re.sub(r'\n+', '\n', data)
        data = re.sub(r'\n\s\s', '\n', data)
        data = re.sub(r'﻿', '', data)
    
    # Для n=1 (одиничні елементи)
    if n == 1:
        if split == "word":
            # Обробка тексту для слів
            if not computer_code:
                data = re.sub(r'--', ' -', data)
            processor = NgrammProcessor(computer_code=computer_code, ignore_comments=ignore_comments)
            processor.preprocess(data, file_name=filename)
            result = processor.get_words()
            L = len(result)
            return result
            
        elif split == 'letter':
            # Обробка для літер і чисел
            temp = []
            data = remove_punctuation(data)
            for word in data:
                for i in word:
                    if i == ' ':
                        continue
                    temp.append(i)
            L = len(temp)
            return temp
            
        elif split == 'symbol':
            # Обробка для символів
            result = []
            for char in data:
                if char == " " or char == "\n" or char == "\ufeff":
                    result.append("space")
                else:
                    result.append(char.lower())
            L = len(result)
            return result
    
    # Для n>1 (n-грами)
    else:
        if split == "word":
            # Обробка для n-грам слів
            data = re.sub(r'--', ' -', data)
            processor = NgrammProcessor()
            processor.preprocess(data)
            words = processor.get_words()
            L = len(words)
            
            # Створюємо n-грами з слів
            result = []
            for i in range(L - n + 1):
                window = tuple(words[i:i + n])
                result.append(window)

            return result
                
        elif split == "letter":
            # Обробка для n-грам літер і чисел
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
            # Обробка для n-грам символів
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