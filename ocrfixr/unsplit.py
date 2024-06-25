import re
import importlib_resources

# Load project resources
ocrfixr = importlib_resources.files("ocrfixr")

# Helper function to load text files
def load_text_file(file_path):
    return (ocrfixr / "data" / file_path).read_text(encoding='utf-8').split()

# Load necessary data
word_set = set(load_text_file("SCOWL_70.txt"))
common_words = set(load_text_file("SCOWL_20.txt"))

class Unsplit:
    def __init__(self, text, return_fixes="F"):
        self.text = text
        self.return_fixes = return_fixes

    def _list_split_words(self):
        # Split text into tokens considering hyphenated words
        tokens = re.split(r" |\n", self.text)
        tokens = [token.strip() for token in tokens]

        # Identify split words
        regex = re.compile(r".+[^-](-\n).+")
        split_words = [token for token in tokens if regex.match(token)]

        return split_words

    def _decide_hyphen(self, text):
        # Remove leading/trailing punctuation and newline hyphen
        no_punct = re.sub(r"^[[:punct:]]+|[[:punct:]]+$", "", text)
        no_hyphen = no_punct.replace("-\n", "")

        # Define word segments
        W0 = no_hyphen
        W1 = re.findall(r".*(?=-\n)", no_punct)[0]
        W2 = re.findall(r"(?<=-\n).*", no_punct)[0]

        # Define possible adjustments
        remove_hyphen = text.replace("-\n", "") + "\n"
        keep_hyphen = text.replace("-\n", "-") + "\n"
        unsure_hyphen = text.replace("-\n", "-*") + "\n"
        end_pg_hyphen = text.replace("-\n", "-*\n") + " "

        # Define tests of "wordiness"
        W0_real = W0 in word_set
        W0_common = W0 in common_words
        W1_real = W1 in word_set
        W2_real = W2 in word_set
        has_proper = W1.istitle() and not W2.istitle()
        has_num = any(char.isdigit() for char in W1) or any(char.isdigit() for char in W2)
        end_pg = "--File" in W2 or W2.isdigit()

        # Decide whether a split word should retain its hyphen
        if end_pg:
            return end_pg_hyphen
        if W0_real:
            if W0_common:
                return remove_hyphen
            if W1_real and W2_real:
                return unsure_hyphen
            return remove_hyphen
        if W1_real and W2_real or has_num or has_proper:
            return keep_hyphen
        return remove_hyphen

    def _multi_replace(self, fixes):
        if not fixes:
            return self.text
        text_corrected = self.text
        for old, new in fixes.items():
            pattern = re.escape(old) + r"(\s|\n)?"
            text_corrected = re.sub(pattern, new, text_corrected)
        return text_corrected

    def _find_replacements(self, splits):
        new_word = [self._decide_hyphen(word) for word in splits]
        return dict(zip(splits, new_word))

    def fix(self):
        splits = self._list_split_words()

        if not splits:
            return [self.text, {}] if self.return_fixes == "T" else self.text

        fixes = self._find_replacements(splits)
        corrected_text = self._multi_replace(fixes)
        return [corrected_text, fixes] if self.return_fixes == "T" else corrected_text

# Example usage:
# text = "Some example text with split-\nwords to be corrected."
# unsplit_instance = Unsplit(text, return_fixes="T")
# result = unsplit_instance.fix()
# print(result)
