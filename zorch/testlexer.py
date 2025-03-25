from enum import Enum

import re


class Token(Enum):
    IDENTIFIER = r"[a-zA-Z_]\w*\b"
    CONSTANT = r"[0-9]+\b"
    INT_KEYWORD = r"int\b"
    VOID_KEYWORD = r"void\b"
    RETURN_KEYWORD = r"return\b"
    OPEN_PARENTHESIS = r"\("
    CLOSE_PARENTHESIS = r"\)"
    OPEN_BRACE = r"{"
    CLOSE_BRACE = r"}"
    SEMICOLON = r";"
    BITWISE_COMPLEMENT_OPERATOR = r"~"
    NEGATION_OPERATOR = r"-"
    DECREMENT_OPERATOR = r"--"
    ADDITION_OPERATOR = r"\+"
    MULTIPLICATION_OPERATOR = r"\*"
    DIVISION_OPERATOR = r"/"
    REMAINDER_OPERATOR = r"%"
    LOGICAL_NOT_OPERATOR = r"!"
    LOGICAL_AND_OPERATOR = r"&&"
    LOGICAL_OR_OPERATOR = r"\|\|"
    EQUAL_TO_OPERATOR = r"=="
    NOT_EQUAL_TO_OPERATOR = r"!="
    LESS_THAN_OPERATOR = r"<"
    GREATER_THAN_OPERATOR = r">"
    LESS_THAN_OR_EQUAL_TO_OPERATOR = r"<="
    GREATER_THAN_OR_EQUAL_TO_OPERATOR = r">="
    ASSIGNMENT_OPERATOR = r"="
    IF_KEYWORD = r"if\b"
    ELSE_KEYWORD = r"else\b"
    QUESTION_MARK_DELIMITER = r"\?"
    COLON_DELIMITER = r":"
    DO_KEYWORD = r"do\b"
    WHILE_KEYWORD = r"while\b"
    FOR_KEYWORD = r"for\b"
    BREAK_KEYWORD = r"break\b"
    CONTINUE_KEYWORD = r"continue\b"
    COMMA = r","
    STATIC_KEYWORD = r"static\b"
    EXTERN_KEYWORD = r"extern\b"


TOKEN_PATTERNS = {t: re.compile(t.value) for t in Token}


def tokenize(s):
    tokens = []
    while s := s.lstrip():
        longest_length = 0
        for pattern in TOKEN_PATTERNS.values():
            if m := pattern.match(s):
                match = m.group()
                match_length = len(match)
                if match_length > longest_length:
                    longest = match
                    longest_length = match_length
        if not longest_length:
            raise RuntimeError(f"No match found for '{s}'")
        tokens.append(longest)
        s = s[longest_length:]
    return tokens

if __name__ == "__main__":
    print(tokenize("int main(void) {return 0;}"))