"""NLP utilities for tokenization, vocabulary building, and data serialization."""

import pickle
from collections import Counter
import re
import csv
from typing import List
import numpy as np

def tokenize(text):
    """Lowercase text and extract alphanumeric word tokens.
    
    Args:
        text: Input string to tokenize.
        
    Returns:
        List of lowercase word tokens.
    """
    text = text.lower()
    return re.findall(r'\b\w+\b', text)


def build_and_save_vocab(corpus_file_path, output_pickle_path, k, max_lines):
    """Build vocabulary from top k frequent words and save to pickle.
    
    Args:
        corpus_file_path: Path to corpus file (UTF-8).
        output_pickle_path: Output pickle file path.
        k: Number of top words to include.
        max_lines: Max lines to process.
        
    Returns:
        Dict mapping words to indices (most frequent = 0).
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if max_lines <= 0:
        raise ValueError(f"max_lines must be positive, got {max_lines}")
    
    word_counts = Counter()

    try:
        with open(corpus_file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break

                tokens = tokenize(line)
                word_counts.update(tokens)
    except Exception as e:
        print(f"Error: {e}")
        raise

    # Build vocab from top k words
    most_common = word_counts.most_common(k)
    vocab = {word: idx for idx, (word, _) in enumerate(most_common)}

    # Save to pickle
    save_pickle(vocab, output_pickle_path)

    return vocab

def load_pickle(pickle_path):
    """Load and deserialize object from pickle file.
    
    Args:
        pickle_path: Path to pickle file.
        
    Returns:
        Deserialized Python object.
    """
    try:
        with open(pickle_path, 'rb') as f:
            vocab = pickle.load(f)
    except Exception as e:
        print(f"Error: {e}")
        raise
    return vocab

def save_pickle(var, output_pickle_path):
    """Serialize and save object to pickle file.
    
    Args:
        var: Object to serialize.
        output_pickle_path: Output file path.
    """
    try:
        with open(output_pickle_path, 'wb') as f:
            pickle.dump(var, f)
    except Exception as e:
        print(f"Error: {e}")
        raise


def read_csv_to_tuples(filepath: str) -> tuple[List[str], np.ndarray]:
    """Parse CSV file and separate into feature and target arrays.
    
    Args:
        filepath: Path to CSV file (UTF-8).
        
    Returns:
        Tuple of (X, y) where X is list of strings and y is NumPy array of floats.
    """
    data = []

    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if len(row) == 2:
                    string_value = row[0].strip()
                    float_value = float(row[1].strip())
                    data.append((string_value, float_value))
    except Exception as e:
        print(f"Error: {e}")
        raise

    # Separate tuples into X (features) and y (targets) for ML compatibility
    X = []
    y = []
    for tup in data:
        X.append(tup[0])
        y.append(tup[1])

    return X, np.array(y)