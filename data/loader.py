import base64
import io
from typing import Optional, Tuple, Dict, Any
import tkinter as tk 
from tkinter import filedialog
import chardet

from models.state_manager import state
 

def detect_encoding(content: bytes) -> str:
    """
    Визначає кодування файлу.
    
    Args:
        content: Байтовий вміст файлу
        
    Returns:
        str: Назва кодування
    """
    result = chardet.detect(content)
    encoding = result.get('encoding', 'utf-8')
    
    # Fallback якщо кодування не визначено
    if encoding is None:
        encoding = 'utf-8'
    
    return encoding


def decode_base64_content(content_string: str) -> bytes:
    """
    Декодує base64 контент з Dash upload.
    
    Args:
        content_string: Рядок у форматі "data:...;base64,..."
        
    Returns:
        bytes: Декодований байтовий вміст
    """
    # Розділяємо заголовок та дані
    if ',' in content_string:
        content_type, content_data = content_string.split(',', 1)
        return base64.b64decode(content_data)
    return base64.b64decode(content_string)


def load_file(content: str, filename: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Завантажує файл з base64 контенту.
    
    Args:
        content: Base64 закодований вміст файлу
        filename: Ім'я файлу
        
    Returns:
        Tuple[Optional[str], Optional[str]]: (вміст файлу, повідомлення про помилку)
    """
    try:
        decoded = decode_base64_content(content)
   
        encoding = detect_encoding(decoded)

        try:
            text = decoded.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            text = decoded.decode('utf-8', errors='ignore')
        
        return text, None
        
    except Exception as e:
        return None, f"Error loading file {filename}: {str(e)}"


def get_file_content(uploaded_files: Dict[str, str], filename: str) -> Optional[str]:
    """
    Отримує вміст файлу зі словника завантажених файлів.
    
    Args:
        uploaded_files: Словник з завантаженими файлами {filename: content}
        filename: Ім'я файлу
        
    Returns:
        Optional[str]: Вміст файлу або None
    """
    if filename in uploaded_files:
        content = uploaded_files[filename]
        text, error = load_file(content, filename)
        if error:
            print(error)
            return None
        return text
    return None


def parse_uploaded_files(contents: list, filenames: list) -> Dict[str, str]:
    """
    Парсить завантажені файли з Dash upload компонента.
    
    Args:
        contents: Список base64 контентів
        filenames: Список імен файлів
        
    Returns:
        Dict[str, str]: Словник {filename: decoded_content}
    """
    files = {}
    
    if contents is None or filenames is None:
        return files
    
    for content, filename in zip(contents, filenames):
        text, error = load_file(content, filename)
        if text is not None:
            files[filename] = text
        else:
            print(f"Warning: Could not load {filename}: {error}")
    
    return files


def calculate_file_lengths(text: str) -> Dict[str, int]:
    """
    Обчислює довжини файлу в різних режимах.
    
    Args:
        text: Вміст файлу
        
    Returns:
        Dict[str, int]: Словник з довжинами {'word': N, 'symbol': N, 'letter': N}
    """
    from .text_processing import remove_punctuation_for_words, remove_punctuation
 
    words = remove_punctuation_for_words(text)
    word_length = len(words)
    
    symbol_length = len(text)
    
    cleaned = remove_punctuation(text)
    letter_length = len([char for char in cleaned if not char.isspace()])
    
    return {
        'word': word_length,
        'symbol': symbol_length,
        'letter': letter_length
    }
    
def pick_folder():
    """Opens folder picker dialog."""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    new_folder = filedialog.askdirectory()
    if new_folder and new_folder is not None and new_folder != "":
        state.save_folder = new_folder
    root.destroy()