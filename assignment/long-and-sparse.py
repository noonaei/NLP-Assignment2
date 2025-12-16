# todo: go over the wikipedia source only the 8M first lines.
# todo: filter the vocabulary to only the 10k most frequent words
import time
import string
from collections import defaultdict


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

