from PySide6.QtCore import QRegularExpression
from PySide6.QtGui import QColor, QFont, QSyntaxHighlighter, QTextCharFormat
from PySide6.QtWidgets import QApplication


def txformat(color, style=""):
    """Return a QTextCharFormat with the given attributes."""
    _format = QTextCharFormat()

    if isinstance(color, str):
        # Handle named colors (fallback)
        _color = QColor()
        _color.setNamedColor(color)
        _format.setForeground(_color)
    else:
        # Assume color is a QColor or palette role
        _format.setForeground(color)

    if "bold" in style:
        _format.setFontWeight(QFont.Bold)
    if "italic" in style:
        _format.setFontItalic(True)

    return _format


STYLES_light = {
    "keyword": txformat("navy", "bold"),
    "defclass": txformat("black", "bold"),
    "string": txformat("green", "bold"),
    "string2": txformat("green"),
    "comment": txformat("darkGray", "italic"),
    "numbers": txformat("brown"),
}

STYLES_dark = {
    "keyword": txformat(QColor(88, 156, 214), "bold"),  # Blue
    "defclass": txformat(QColor(78, 201, 176), "bold"),  # Teal
    "string": txformat(QColor(106, 153, 85)),
    "string2": txformat(QColor(106, 153, 85)),
    "comment": txformat(QColor(181, 206, 168), "italic"),  # Grey-green
    "numbers": txformat(QColor(206, 145, 120)),  # orange
}


def is_dark():
    app = QApplication.instance()
    if hasattr(app, "styleHints") and hasattr(app.styleHints(), "colorScheme"):
        return app.styleHints().colorScheme().value == 2
    return False


class PythonHighlighter(QSyntaxHighlighter):
    """Syntax highlighter for the Python language."""

    # Python keywords
    keywords = [
        "and",
        "assert",
        "break",
        "class",
        "continue",
        "def",
        "del",
        "elif",
        "else",
        "except",
        "exec",
        "finally",
        "for",
        "from",
        "global",
        "if",
        "import",
        "in",
        "is",
        "lambda",
        "not",
        "or",
        "pass",
        "print",
        "raise",
        "return",
        "try",
        "while",
        "yield",
        "None",
        "True",
        "False",
        "as",
    ]

    # noinspection PyArgumentList
    def __init__(self, document):
        super().__init__(document)

        STYLES = STYLES_dark if is_dark() else STYLES_light

        # Multi-line strings (expression, flag, style)
        # FIXME: The triple-quotes in these two lines will mess up the
        # syntax highlighting from this point onward
        self.tri_single = (QRegularExpression("'''"), 1, STYLES["string2"])
        self.tri_double = (QRegularExpression('"""'), 2, STYLES["string2"])

        rules = []

        # Keyword, operator, and brace rules
        rules += [(r"\b%s\b" % w, 0, STYLES["keyword"]) for w in PythonHighlighter.keywords]

        # All other rules
        rules += [
            # Double-quoted string, possibly containing escape sequences
            (r'"[^"\\]*(\\.[^"\\]*)*"', 0, STYLES["string"]),
            # Single-quoted string, possibly containing escape sequences
            (r"'[^'\\]*(\\.[^'\\]*)*'", 0, STYLES["string"]),
            # 'def' followed by an identifier
            (r"\bdef\b\s*(\w+)", 1, STYLES["defclass"]),
            # 'class' followed by an identifier
            (r"\bclass\b\s*(\w+)", 1, STYLES["defclass"]),
            # From '#' until a newline
            (r"#[^\n]*", 0, STYLES["comment"]),
            # Numeric literals
            (r"\b[+-]?[0-9]+[lL]?\b", 0, STYLES["numbers"]),
            (r"\b[+-]?0[xX][0-9A-Fa-f]+[lL]?\b", 0, STYLES["numbers"]),
            (r"\b[+-]?[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\b", 0, STYLES["numbers"]),
        ]

        # Build a QRegularExpression for each pattern
        # noinspection PyArgumentList
        self.rules = [(QRegularExpression(pat), index, fmt) for (pat, index, fmt) in rules]

    def highlightBlock(self, text):
        """Apply syntax highlighting to the given block of text."""
        # Do other syntax formatting
        for expression, nth, _format in self.rules:
            match = expression.match(text)
            index = match.capturedStart()

            while index >= 0:
                # We actually want the index of the nth match
                length = match.capturedLength()
                self.setFormat(index, length, _format)
                match = expression.match(text, index + length)
                index = match.capturedStart()

        self.setCurrentBlockState(0)

        # Do multi-line strings
        in_multiline = self.match_multiline(text, *self.tri_single)
        if not in_multiline:
            self.match_multiline(text, *self.tri_double)

    def match_multiline(self, text, delimiter, in_state, style):
        """Do highlighting of multi-line strings. ``delimiter`` should be a
        ``QRegularExpression`` for triple-single-quotes or triple-double-quotes, and
        ``in_state`` should be a unique integer to represent the corresponding
        state changes when inside those strings. Returns True if we're still
        inside a multi-line string when this function is finished.
        """
        # If inside triple-single quotes, start at 0
        if self.previousBlockState() == in_state:
            start = 0
            add = 0
        # Otherwise, look for the delimiter on this line
        else:
            match = delimiter.match(text)
            start = match.capturedStart()
            add = match.capturedLength()

        # As long as there's a delimiter match on this line...
        while start >= 0:
            # Look for the ending delimiter
            match = delimiter.match(text, start + add)
            end = match.capturedStart()
            # Ending delimiter on this line?
            if end >= add:
                length = end - start + add + match.capturedLength()
                self.setCurrentBlockState(0)
            # No; multi-line string
            else:
                self.setCurrentBlockState(in_state)
                length = len(text) - start + add
            # Apply formatting
            self.setFormat(start, length, style)
            # Look for the next match
            match = delimiter.match(text, start + length)
            start = match.capturedStart()

        # Return True if still inside a multi-line string, False otherwise
        if self.currentBlockState() == in_state:
            return True
        else:
            return False
