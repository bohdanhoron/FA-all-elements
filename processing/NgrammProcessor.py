import re
from string import punctuation
from typing import List, Tuple, Optional, Dict, Any, Union

from .CodeTokenizer import CodeTokenizer

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

        if not self.computer_code:
            regexs = [(r'\n+', '\n'), (r'\n\s+', '\n'), (r'<feff>', ''), (r'--', ' -'), (r'\n', ' ')]

            for regex in regexs:
                text = re.sub(regex[0], regex[1], text)
            
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


