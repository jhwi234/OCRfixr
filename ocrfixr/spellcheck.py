from transformers import logging, pipeline
import re
import string
import ast
import importlib_resources
from collections import Counter
from symspellpy import SymSpell, Verbosity
from metaphone import doublemetaphone
import pkg_resources

# Suppress transformers logging
logging.set_verbosity_error()

# Load project resources
ocrfixr = importlib_resources.files("ocrfixr")

# Helper function to load text files
def load_text_file(file_path):
    return (ocrfixr / "data" / file_path).read_text(encoding='utf-8').split()

# Load necessary data
word_set = set(load_text_file("SCOWL_70.txt"))
ignore_set_from_pkg = set(load_text_file("Ignore_These_Misspells.txt"))

def load_dict_file(file_path):
    return ast.literal_eval((ocrfixr / "data" / file_path).read_text(encoding='utf-8'))

common_scannos = load_dict_file("Scannos_Common.txt")
stealth_scannos = load_dict_file("Scannos_Stealth.txt")
ignore_suggestions = load_dict_file("Ignore_These_Suggestions.txt")

# Initialize symspell
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

# Initialize BERT model
unmasker = pipeline('fill-mask', model='bert-base-uncased', top_k=30)

class SpellCheck:
    def __init__(self, text, changes_by_paragraph="F", return_fixes="F", ignore_words=None,
                 interactive="F", common_scannos="T", top_k=15, return_context="F", suggest_unsplit="T"):
        self.text = text
        self.changes_by_paragraph = changes_by_paragraph
        self.return_fixes = return_fixes
        self.ignore_words = ignore_words or []
        self.interactive = interactive
        self.common_scannos = common_scannos
        self.top_k = top_k
        self.return_context = return_context
        self.suggest_unsplit = suggest_unsplit

    def _split_paragraphs(self, text):
        return re.findall('[^\n]+\n{0,}|(?:\w+\s+[^\n]){500}', text)

    def _list_misreads(self):
        tokens = re.split("[ \n]", self.text)
        tokens = [l.strip() for l in tokens]

        no_hyphens = re.compile(".*-.*|.*'.*|.*\.{3,}.*|.*’.*|[0-9]+")
        no_caps = re.compile('[^A-Z]{2,}')
        no_footnotes = re.compile('.*[0-9]{1,}[^A-z]?$')
        no_roman_nums = re.compile('[xlcviXLCVI.:,-;]+$')
        no_eth_endings = re.compile('.*eth|.*est$')
        no_format_tags = re.compile('</?[a-z]>.*|.*</?[a-z]>')
        no_list_items = re.compile('.*\\)|.*\\]')
        all_nums = re.compile('^[0-9]{1,}$')

        words = [x for x in tokens if not no_hyphens.match(x) and no_caps.match(x) and 
                 not no_footnotes.match(x) and not no_roman_nums.match(x) and 
                 not no_eth_endings.match(x) and not no_format_tags.match(x) and 
                 not no_list_items.match(x)]

        no_punctuation = [l.strip(string.punctuation+"”“’‘") for l in words]
        words_to_check = [x for x in no_punctuation if len(x) > 1 and not all_nums.match(x)]

        unrecognized = [i for i in words_to_check if i not in word_set]

        if len(tokens) > 10 and len(unrecognized) / len(tokens) > 0.30:
            return []
        else:
            ignore_set_from_user = set(self.ignore_words)
            misread = [i for i in unrecognized if i not in ignore_set_from_pkg and i not in ignore_set_from_user]

            if self.common_scannos == "T":
                for i in tokens:
                    if i not in misread and i in common_scannos or i in stealth_scannos:
                        misread.append(i)

            return misread

    def _count_misreads(self):
        all_misreads = Counter(self._list_misreads())
        multi_misreads = {k: v for k, v in all_misreads.items() if v > 2}
        return multi_misreads

    def _suggest_spellcheck(self, text):
        suggested_words = []

        num_spaces = [getattr(i, "term") for i in sym_spell.lookup_compound(text, max_edit_distance=0)]

        if str(num_spaces).count(' ') == 0:
            suggested_words.extend(
                [getattr(i, "term") for i in sym_spell.lookup(text, Verbosity.CLOSEST, max_edit_distance=2) if len(getattr(i, "term")) > 1]
            )
        else:
            mw = num_spaces.pop()
            if "," in text:
                mw = re.sub(" ", ", ", mw)
            suggested_words.append(mw)

        return suggested_words

    def _suggest_bert(self, text, number_to_return=15):
        context_suggest = unmasker(text)
        suggested_words = [x.get("token_str") for x in context_suggest][:number_to_return]
        return suggested_words

    def _list_to_str(self, list_items):
        return ' '.join(map(str, list_items))

    def _set_mask(self, orig_word, replacement, orig_text):
        return orig_text.replace(str(orig_word), str(replacement), 1)

    def _multi_replace(self, fixes):
        if not fixes:
            return self.text
        text_corrected = self.text
        for i, j in fixes.items():
            pattern = "\\b" + re.escape(i) + "\\b"
            text_corrected = re.sub(pattern, j, text_corrected)
        return text_corrected

    def _find_replacements(self, misreads):
        sc = []
        bert = []
        punct_split_fixes = {}
        common_scanno_fixes = {}

        for i in misreads:
            if self.common_scannos == "T" and i in common_scannos:
                common_scanno_fixes.update({i: common_scannos[i]})
            elif self.common_scannos == "T" and i in stealth_scannos:
                sc.append(stealth_scannos.get(i).split(" "))
                sb = self._suggest_bert(self._set_mask(i, '[MASK]', self.text), self.top_k)
                if i not in sb:
                    bert.append(sb)
            else:
                spellcheck = self._suggest_spellcheck(i)
                if spellcheck == i:
                    misreads.remove(i)
                else:
                    if self.suggest_unsplit == "T" and len(spellcheck) == 1 and str(spellcheck).count(' ') == 1:
                        if "." in i or "," in i:
                            mw = ''.join(spellcheck)
                            fw, sw = re.findall("^[^\s,]+", mw).pop(), re.findall("[^\s]+$", mw).pop()
                            if fw in word_set and sw in word_set:
                                if len(re.findall("\.{1,}([A-Z][a-z]+)", i)) > 0:
                                    fw += '. '
                                    sw = str.title(sw)
                                    mw = fw + sw
                                punct_split_fixes.update({i: mw})
                        else:
                            sc.append(spellcheck)
                            mw = ''.join(spellcheck)
                            fw = re.findall("^[^\s]+", mw).pop()
                            sb = self._suggest_bert(self._set_mask(i, fw + ' [MASK]', self.text), self.top_k)
                            sbi = [fw + ' ' + x for x in sb]
                            bert.append(sbi)
                    else:
                        sc.append(spellcheck)
                        bert.append(self._suggest_bert(self._set_mask(i, '[MASK]', self.text), self.top_k))

        corr = []
        fixes = []
        x = 0
        while x < len(bert):
            overlap = set(bert[x]) & set(sc[x])
            corr.append(overlap)
            corr[x] = self._list_to_str(corr[x]) if len(overlap) == 1 else ""
            x += 1

        fixes = dict(zip(misreads, corr))

        try:
            for key, value in fixes.copy().items():
                if value + "s" == key:
                    del fixes[key]
                elif key[len(key) - 1] == "o" and value[len(value) - 1] == "e":
                    pass
                elif value.count(' ') == 1:
                    pass
                elif doublemetaphone(key)[0] == doublemetaphone(value)[0]:
                    del fixes[key]
        except Exception:
            pass

        fixes.update(common_scanno_fixes)
        fixes.update(punct_split_fixes)

        no_change = [k for k, v in fixes.items() if k == v]
        for x in no_change:
            del fixes[x]

        overlap = dict(fixes.items() & ignore_suggestions.items())
        for x in overlap:
            del fixes[x]

        if self.interactive == "T":
            for key in list(fixes.keys()):
                if fixes[key] == "":
                    del fixes[key]
            for key, value in fixes.items():
                self._create_dialogue(self.text, key, value)
                if not proceed:
                    fixes.update({key: ""})

        for key in list(fixes.keys()):
            if fixes[key] == "":
                del fixes[key]

        return fixes

    def single_string_fix(self):
        misreads = self._list_misreads()
        if not misreads:
            return [] if self.changes_by_paragraph == "T" else [self.text, {}]

        fixes = self._find_replacements(misreads)
        correction = self._multi_replace(fixes)
        if self.changes_by_paragraph == "T":
            if not fixes:
                return []
            full_results = []
            for key, value in fixes.items():
                loc = self.text.find(key) - 1
                context_text = '{0} Suggest \'{1}\' for \'{2}\''.format(loc, value, key)
                if self.return_context == "T":
                    full_results.append(context_text + ' | ' + self.text)
                else:
                    full_results.append(context_text)
            return full_results
        else:
            return [correction, fixes]

    def fix(self):
        open_list = [spellcheck(paragraph, changes_by_paragraph=self.changes_by_paragraph,
                                interactive=self.interactive, common_scannos=self.common_scannos,
                                top_k=self.top_k, return_context=self.return_context,
                                suggest_unsplit=self.suggest_unsplit).single_string_fix()
                     for paragraph in self._split_paragraphs(self.text)]

        if self.changes_by_paragraph == "T":
            open_list = list(filter(None, open_list))
            if not open_list:
                return "NOTE: No changes made to text"
            return '\n'.join(sum(open_list, []))
        else:
            corrections = [x[0] for x in open_list]
            final_text = ''.join(corrections)
            if self.return_fixes == "T":
                fixes = [x[1] for x in open_list]
                word_changes = list(j for i in fixes for j in i.items())
                counts = dict(Counter(word_changes))
                counts = dict(sorted(counts.items(), key=lambda item: -item[1]))
                return [final_text, counts]
            return final_text

# TODO - (ADD_DICTS) Need to add selectable foreign language dictionaries
# TODO - (IGNORE_SPLIT_WORDS) need to ignore the first word of a new page, since these can be split words across pages (this may also just be tied up in the unsplit functionality, where this word should have a leading * to denote a split word)
# TODO - (ADD_STEALTHOS) Need to add additional common stealth scannos to OCRfixr. Be mindful, as these can increase compute time hugely (eg. he/be). Shoot for words that are uncommon (arid --> and)
# TODO - (FULL_PARAGRAPHS) Allow BERT context to draw from all lines in a full paragraph (currently resets at each newline -- this corresponds to 1 line of text in a Gutenberg text, and likely leads to degraded spellcheck performance due to loss of context). However, longer context window = slower performance
#          > most useful case for this is when the MASKED word is the first or last word in the line
#          > exploring the option of accepting synonyms for context words as valid spellcheck replacements (ex. "the dark, [MASK] swamp". wet --> damp)
# TODO - (GutenBERT) fine-tune BERT model on Gutenberg texts, to improve relatedness of context suggestions
# TODO - (WARM_UP) can we somehow negate the warm-up time for the transformers unmasker?
# pipelines = 7 secs
# symspellpy dictionary load = 3 seconds
