"""Word valence prediction using co-occurrence matrix representations.

Builds word embeddings from co-occurrence statistics and trains a linear regression
model to predict valence scores for words based on their distributional context.
"""

from utils import *
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error
import numpy as np


MAX_LINES = 7_000_000  # 7 million lines
TOP_K = 10000
CORPUS_FILE_NAME = './data/en.wikipedia2018.10M.txt'
WINDOW_SIZE = 3
VOCAB_PICKLE_FILE = './pkl/vocab.pkl'


def build_co_occurrence_matrix(corpus_file_name, data: list, vocab, window_size, max_lines=MAX_LINES):
    """Build co-occurrence matrix for target words within a context window.
    
    Constructs a matrix where each row represents a target word and columns
    represent context words from the vocabulary. Values are co-occurrence counts
    within the specified window size.
    
    Args:
        corpus_file_name: Path to corpus text file.
        data: List of target words to build representations for.
        vocab: Dict mapping vocabulary words to column indices.
        window_size: Number of words to consider on each side of target.
        max_lines: Maximum corpus lines to process.
        
    Returns:
        NumPy array of shape (len(data), len(vocab)) with co-occurrence counts.
    """
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if max_lines <= 0:
        raise ValueError(f"max_lines must be positive, got {max_lines}")
    if not data:
        raise ValueError("data list cannot be empty")
    if not vocab:
        raise ValueError("vocab dict cannot be empty")
    
    # Map each target word to its row index in the matrix
    data_to_idx = {word: i for i, word in enumerate(data)}
    # Use set for O(1) lookup when checking if word is a target
    data_lookup = set(data)

    # Initialize matrix as list of lists (faster for incremental updates than numpy)
    mat =  [[0 for _ in range(len(vocab))] for _ in range(len(data))]

    line_idx = 0
    try:
        with open(corpus_file_name, 'r', encoding='utf-8') as fin:
            for line in fin:
                if line_idx >= max_lines:
                    break
                line_idx += 1
                if line_idx % 100000 == 0:
                    print(f"Processed line: {line_idx}")

                cleaned_line = tokenize(line)
                n = len(cleaned_line)

                for idx, word in enumerate(cleaned_line):
                    if word in data_lookup:
                        # Calculate window boundaries, ensuring we don't go out of bounds
                        start = max(0, idx - window_size)
                        end = min(n, idx + window_size + 1)

                        # Left context
                        for j in range(start, idx):
                            context_word = cleaned_line[j]
                            if context_word in vocab:
                                mat[data_to_idx[word]][vocab[context_word]] += 1

                        # Right context
                        for j in range(idx + 1, end):
                            context_word = cleaned_line[j]
                            if context_word in vocab:
                                mat[data_to_idx[word]][vocab[context_word]] += 1

    except Exception as e:
        print(f"Error: {e}")
        raise
    return np.array(mat)