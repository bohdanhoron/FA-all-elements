import base64
import chardet
import re
from typing import List, Dict, Tuple
import dash_html_components as html

from models.state_manager import state_manager
from data.text_processing import NgrammProcessor, remove_punctuation

def decode_file_content(content_string: str) -> Tuple[str, str]:
    decoded = base64.b64decode(content_string)
    detection = chardet.detect(decoded)
    encoding = detection['encoding'] or 'windows-1251'
    return decoded.decode(encoding), encoding

def calculate_initial_lengths(file_content: str, filename: str, processor_mode: str, ignore_comments: bool) -> Dict[str, int]:
    lengths = {}
    computer_code = (processor_mode == 'computer_code')
    
    text_word = file_content
    if not computer_code:
        text_word = re.sub(r'\n+', '\n', file_content)
        text_word = re.sub(r'\n\s\s', '\n', text_word)
        text_word = re.sub(r'﻿', '', text_word)
        text_word = re.sub(r'--', ' -', text_word)
    
    processor = NgrammProcessor(computer_code=computer_code, ignore_comments=ignore_comments)
    processor.preprocess(text_word, file_name=filename)
    words = processor.get_words()
    lengths['word'] = len(words)
    
    symbols = []
    for char in file_content:
        if char in [" ", "\n", "\ufeff"]:
            symbols.append("space")
        else:
            symbols.append(char.lower())
    lengths['symbol'] = len(symbols)
    
    text_letter = remove_punctuation(file_content)
    letters = [char for word in text_letter for char in word if char != ' ']
    lengths['letter'] = len(letters)
    
    return lengths

def load_files_to_state(contents: List[str], filenames: List[str], processor_mode: str, ignore_comments: bool) -> Tuple[int, int]:
    success_count = 0
    error_count = 0
    
    for content, filename in zip(contents, filenames):
        try:
            _, content_string = content.split(',')
            file_text, _ = decode_file_content(content_string)
            
            state_manager.uploaded_files[filename] = file_text
            state_manager.file_lengths[filename] = calculate_initial_lengths(
                file_text, filename, processor_mode, ignore_comments
            )
            
            success_count += 1
        except Exception as e:
            print(f"✗ Error processing {filename}: {str(e)}")
            error_count += 1
            
    return success_count, error_count

def get_upload_summary_layout(success_count: int, error_count: int):
    return html.Div([
        html.H5("Upload Summary:"),
        html.P(f"Successfully uploaded: {success_count} file(s)", style={'color': 'green'}),
        html.P(f"Files with errors: {error_count}", style={'color': 'red' if error_count > 0 else 'green'})
    ])