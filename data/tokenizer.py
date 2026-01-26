from pygments.filter import Filter
from pygments.token import Comment, Token
from pygments import lex
from pygments.lexers import guess_lexer_for_filename, get_lexer_by_name
from pygments.token import Token
import re

class CodeTokenizer:
    def __init__(self, code, file_name):
        self.code = code
        self.file_path = file_name


    def process(self):
        try:
            lexer = guess_lexer_for_filename(self.file_path, self.code)
        except Exception:
            lexer = get_lexer_by_name("text")

        lexer.add_filter(UniversalCommentDelimiterSplit())

        self.code = re.sub(r'\\u[0-9A-Fa-f]{4}', lambda m: chr(int(m.group(0)[2:], 16)), self.code)
        tokens = lex(self.code, lexer)

        result = [{'type': str(token_type).replace("Token.", ""), 'value': value} for token_type, value in tokens if token_type not in Token.Text]

        pattern = re.compile(r"=\s*(\d+)((?:_\d+)+);")

        #Об'єднює числа у форматі з нижнім підкресленням 1_000, 1_123_222 в один токен
        match = pattern.search(self.code)
        if match:
            groups = match.groups()
            values_to_match = [groups[0], groups[1]]

            for value in values_to_match:
                token = next((t for t in result if t['value'] == value), None)
                if token:
                    result.remove(token)

            combined_value = ''.join(values_to_match)
            result.append({
                "type": "Literal.Number.Integer",
                "value": combined_value
            })
        return result


class UniversalCommentDelimiterSplit(Filter):

    def __init__(self, **options):
        super().__init__(**options)

    COMMENT_PATTERNS = [
        (re.compile(r'^(/\*)(.*?)(\*/)$', re.DOTALL), '/*', '*/'),  # C-style
        (re.compile(r'^(<!--)(.*?)(-->)$', re.DOTALL), '<!--', '-->'),  # HTML
        (re.compile(r'^(--)(.*)', re.DOTALL), '--', ''),  # SQL/Lua single-line
        (re.compile(r'^(//)(.*)', re.DOTALL), '//', ''),  # C++/Java single-line
        (re.compile(r'^(#)(.*)', re.DOTALL), '#', ''),  # Python/Bash/YAML
        (re.compile(r"^(''')(.*)(''')$", re.DOTALL), "'''", "'''"),  # Python triple
        (re.compile(r'^(--\[\[)(.*)(\]\])$', re.DOTALL), '--[[', ']]'),  # Lua multi-line
    ]

    def filter(self, lexer, stream):
        for ttype, value in stream:
            if ttype in Comment:
                for pattern, start, end in self.COMMENT_PATTERNS:
                    m = pattern.match(value)
                    if m:
                        if start:
                            yield (Token.Comment.Special, m.group(1))
                        if m.lastindex >= 2 and m.group(2):
                            yield (ttype, m.group(2))
                        if m.lastindex == 3 and end:
                            yield (Token.Comment.Special, m.group(3))
                        break
                else:
                    yield (ttype, value)
            else:
                yield (ttype, value)
