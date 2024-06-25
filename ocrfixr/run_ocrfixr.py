#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from transformers import logging
logging.set_verbosity_error()
import sys
import re
from tqdm import tqdm
from collections import Counter
from ocrfixr import unsplit, spellcheck

def load_text(file_path):
    """Load text from a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def save_text(file_path, text):
    """Save text to a file."""
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(text)

def count_misspells(text, min_len):
    """Count misspelled words in the text."""
    misreads = spellcheck(text)._LIST_MISREADS()
    filtered_misreads = [word for word in misreads if len(word) > min_len]
    counts = dict(Counter(filtered_misreads))
    return dict(sorted(counts.items(), key=lambda item: -item[1]))

def handle_split_words(text):
    """Handle split words in the text."""
    if len(re.findall("[A-z]-\n", text)) > 30:
        print("---- This file appears to have words split across lines, which can cause issues with the spellchecker")
        print("---- Merging split words back together...")
        fixed_text = unsplit(text).fix()
        return fixed_text.split("\n")
    else:
        return text.split("\n")

def get_ignored_words(full_text, min_len):
    """Get words to be ignored if they appear more than 10 times."""
    counts = count_misspells(full_text, min_len)
    over_ten = {key: value for key, value in counts.items() if value >= 10}
    print("---- To speed things up, OCRfixr will ignore the following unrecognized words that popped up 10 or more times in the text:")
    if len(over_ten) == 0:
        print("NO WORDS IGNORED!")
    else:
        for k, v in over_ten.items():
            print(f"{k} --> {v}")
    return list(over_ten.keys())

def process_lines(data, ignored_words, context_fl):
    """Run spellcheck on each line and collect suggestions."""
    suggestions = []
    for i in tqdm(data):
        fixes = spellcheck(i, changes_by_paragraph="T", return_context=context_fl, ignore_words=ignored_words).fix()
        if fixes != "NOTE: No changes made to text":
            for x in fixes.split("\n"):
                line_number = ' '.join(re.findall('^[0-9]+:', i))
                suggestions.append(f"{line_number} {x}")
    return suggestions

def main():
    parser = argparse.ArgumentParser(prog='run_ocrfixr',
                                     description='Provides context-based spellcheck suggestions for input text.')
    
    parser.add_argument('text', help='path to text you want to spellcheck')
    parser.add_argument('outfile', help='path to output file')
    parser.add_argument('-Warp10', action='store_true', help="option to ignore the most common misspells, which are likely correct words.")
    parser.add_argument('-context', action='store_true', help="option to add local context of suggested change.")
    parser.add_argument('-misspells', action='store_true', help="option to return all of the words OCRfixr didn't recognize.")
    
    args = parser.parse_args()

    # Read and process the text
    print("---- Loading text....")
    full_text = load_text(args.text)
    data = handle_split_words(full_text)

    # Add line numbers
    data = [f'{number + 1}:  {line}' for number, line in enumerate(data)]

    # Handle misspells option
    if args.misspells:
        counts = count_misspells(full_text, 0)
        save_text(args.outfile, '\n'.join(f'{key}:{value}' for key, value in counts.items()))
        print(f"---- File has been written to {args.outfile}")
        sys.exit()

    # Handle Warp10 option
    ignored_words = get_ignored_words(full_text, 3) if args.Warp10 else []

    context_flag = "T" if args.context else "F"

    # Run spellcheck and collect suggestions
    print("---- Running spellcheck....")
    suggestions = process_lines(data, ignored_words, context_flag)

    # Output results
    save_text(args.outfile, '\n'.join(suggestions))
    print(f"---- File has been written to {args.outfile}")

if __name__ == "__main__":
    main()
