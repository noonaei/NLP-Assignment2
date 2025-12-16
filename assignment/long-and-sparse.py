# todo: go over the wikipedia source only the 8M first lines.
# todo: filter the vocabulary to only the 10k most frequent words
import time
import string
from collections import defaultdict
from operator import indexOf

import numpy as np


MAX_LINES = 8000000  # 8 million lines
TOP_K = 10000
CORPUS_FILE_NAME = './data/en.wikipedia2018.10M.txt'

def clean_line(text_line: str) -> str:
    """Cleans a line of text by lowercasing and removing punctuation.

    Args:
        text_line (str): The raw input text line.

    Returns:
        str: The cleaned text line.
    """
    text_line = text_line.lower()
    translator = str.maketrans('', '', string.punctuation)
    cleaned_line = text_line.translate(translator)
    return cleaned_line


def get_word2feq(corpus_filename, max_lines):
    """Builds word frequency dictionary from corpus file.
    
    Args:
        corpus_filename: Path to the corpus text file.
        max_lines: Maximum number of lines to process.
    
    Returns:
        dict: Dictionary mapping words to their frequencies.
    """
    start_time = time.time()
    word2freq = defaultdict(int)
    line_count = 0

    try:
        with open(corpus_filename, 'r', encoding='utf-8') as fin:
            # Iterate over the file line by line
            for line in fin:
                if line_count < max_lines:
                    cleaned_line = clean_line(line)
                    line_count += 1
                    for word in cleaned_line.split():
                        word2freq[word] += 1
                else:
                    # Once the limit is reached, stop reading the file
                    break

        # 3. Output results
        end_time = time.time()
        print(f"Execution time: {end_time - start_time:.4f} seconds")
        print(f"Successfully read and processed the first {line_count} lines of '{corpus_filename}'")
        print(f"Total Unique Words (Vocabulary Size): {len(word2freq)}")

        return word2freq

    except FileNotFoundError:
        print(f"ERROR: The file '{corpus_filename}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred during file processing: {e}")



def get_vocab(word2freq, vocab_size):
    """Extracts the top K most frequent words from a frequency dictionary.
    
    Args:
        word2freq: Dictionary mapping words to their frequencies.
        vocab_size: Number of top words to return.
    
    Returns:
        Set of the top K most frequent words.
    """
    word2freq_len = len(word2freq)
    if word2freq_len < vocab_size:
        # If we have fewer words than requested, return all words as a set
        return set(word2freq.keys())
    else:
        # Get top K most frequent words and return as a set
        top_words = sorted(
            word2freq.items(),
            key=lambda item: item[1],
            reverse=True
        )[:vocab_size]
        # Extract only the words w/o the frequencies
        return {word for word, freq in top_words}


def create_co_occurrence_mat(corpus_file_name, data:list, vocab:list, window_size):
    nrows = len(data)
    ncols = len(vocab)

    data_lookup = set(data)  # Set for O(1) membership checking
    data_lookup_dict = {word: idx for idx, word in enumerate(data)}  # O(1) word-to-row-index lookup
    vocab_lookup = {word: idx for idx, word in enumerate(vocab)}  # O(1) word-to-column-index lookup
    co_occurrence_mat = np.zeros((nrows, ncols))

    # go over the corpus
    # when hit a word that is in the data - check which words accompanying it
    with open(corpus_file_name, 'r', encoding='utf-8') as fin:
        # Iterate over the file line by line
        for line in fin:
            cleaned_line = clean_line(line).split() # todo: when reading corpus the first time should create a cleaned corpus pickle file for faster I/O
            cleaned_line_len = len(cleaned_line)

            for index, word in enumerate(cleaned_line):
                if word in data_lookup: # if we hit a word which is in the data then increment the words accompanying it

                    # increment the words in the left window
                    word_index = data_lookup_dict[word]
                    for i in range(index-window_size, index):
                        if i >= 0:
                            context_word = cleaned_line[i]
                            if context_word in vocab_lookup:
                                co_occurrence_mat[word_index, vocab_lookup[context_word]] += 1

                    # increment the words in the right window
                    for i in range(index+1, index+window_size+1):
                        if i >= cleaned_line_len:
                            break
                        else:
                            context_word = cleaned_line[i]
                            if context_word in vocab_lookup:
                                co_occurrence_mat[word_index, vocab_lookup[context_word]] += 1

    return co_occurrence_mat
