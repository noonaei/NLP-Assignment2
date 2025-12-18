import time
import string
from collections import defaultdict
from csv_reader import *
from dev_utils import *
from collections import Counter
import re
import pickle
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.linear_model import LinearRegression
import numpy as np


MAX_LINES = 7_000_000  # 7 million lines
TOP_K = 10000
CORPUS_FILE_NAME = './data/en.wikipedia2018.10M.txt'
VOCAB_PICKLE_FILE = './vocab.pkl'
PROGRESS_INTERVAL = 100000  # Print progress every N lines


def tokenize(text):
    """Simple tokenizer to clean and split text."""
    text = text.lower()
    return re.findall(r'\b\w+\b', text)


def build_and_save_vocab(corpus_file_path, output_pickle_path, k=TOP_K, max_lines=MAX_LINES):
    """
    Build vocabulary of top k most frequent words and save to pickle file.
    
    Returns: vocab dict {word: idx}
    """
    print(f"[Build Vocab] Starting: Finding top {k} words...")
    
    word_counts = Counter()
    
    with open(corpus_file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            
            if i % PROGRESS_INTERVAL == 0:
                print(f"  Processing line {i:,} / {max_lines:,}")
            
            tokens = tokenize(line)
            word_counts.update(tokens)
    
    # Build vocab from top k words
    most_common = word_counts.most_common(k)
    vocab = {word: idx for idx, (word, _) in enumerate(most_common)}
    
    # Save to pickle
    with open(output_pickle_path, 'wb') as pf:
        pickle.dump(vocab, pf)
    
    print(f"[Build Vocab] Completed: {len(vocab)} words saved to {output_pickle_path}")
    return vocab


def load_vocab(pickle_path):
    """Load vocabulary from pickle file."""
    print(f"[Load Vocab] Loading from {pickle_path}...")
    with open(pickle_path, 'rb') as f:
        vocab = pickle.load(f)
    print(f"[Load Vocab] Loaded {len(vocab)} words")
    return vocab


import numpy as np
from collections import Counter


def build_co_occurrence_matrix(corpus_file_name, data: list, vocab, window_size, max_lines=MAX_LINES):
    t_start = time.time()
    print("Creating co-occurrence matrix (Optimized)...")

    # 1. Pre-compute mappings (O(1) lookups)
    data_to_idx = {word: i for i, word in enumerate(data)}
    vocab_lookup = set(vocab.keys())
    data_lookup = set(data)

    mat =  [[0 for _ in range(len(vocab_lookup))] for _ in range(len(data))]


    line_idx = 0
    with open(corpus_file_name, 'r', encoding='utf-8') as fin:
        for line in fin:
            if line_idx > max_lines:
                break
            line_idx += 1
            if line_idx % 100000 == 0:
                print(f"Processing line: {line_idx}")

            cleaned_line = tokenize(line)
            n = len(cleaned_line)

            for i, word in enumerate(cleaned_line):
                if word in data_lookup:
                    start = max(0, i - window_size)
                    # Right window: from i+1 to min(n, i+window+1)
                    end = min(n, i + window_size + 1)

                    # Combine ranges to iterate purely over neighbors
                    # We use line_vocab_ids because we only care if neighbors are in 'vocab'

                    # Left context
                    for j in range(start, i):
                        context_word = cleaned_line[j]
                        if context_word in vocab:
                            mat[data_to_idx[word]][vocab[context_word]] += 1

                    # Right context
                    for j in range(i + 1, end):
                        context_word = cleaned_line[j]
                        if context_word in vocab:
                            mat[data_to_idx[word]][vocab[context_word]] += 1
        t_end = time.time()

        print(f"buildin the matrix took {t_end - t_start} seconds")
    return np.array(mat)





# ============= STEP 1: Build and save vocab (run once) =============
# Uncomment to build vocab:
# start = time.time()
# vocab = build_and_save_vocab(CORPUS_FILE_NAME, VOCAB_PICKLE_FILE, TOP_K, MAX_LINES)
# print(f"Vocab build took {time.time() - start:.2f} seconds")


# ============= STEP 2: Load vocab and build matrix =============
start = time.time()


#build_and_save_vocab(CORPUS_FILE_NAME, "vocab.pkl", TOP_K, MAX_LINES)
# Load vocab from pickle
vocab = load_vocab(VOCAB_PICKLE_FILE)

# Load target words
train_data = read_csv_to_tuples('./data/nrc-valence-scores.trn.csv')
train_target_words = []
y_train = []
for tup in train_data:
    train_target_words.append(tup[0])
    y_train.append(tup[1])

test_data = read_csv_to_tuples('./data/nrc-valence-scores.tst.csv')
test_target_words = []
y_test = []
for tup in test_data:
    test_target_words.append(tup[0])
    y_test.append(tup[1])

data_for_co_occurrence_mat = train_target_words + test_target_words

# Build matrix
mate = build_co_occurrence_matrix(CORPUS_FILE_NAME, data_for_co_occurrence_mat, vocab,  3,  7_000_000)

# Save to pickle
"""
with open('./pickle_mat.pkl', 'wb') as pmf:
    pickle.dump(vocab, pmf)




reg = LinearRegression().fit(mate, y_train)
reg.score(mate, y_train)


end = time.time()
"""

#print(f"Matrix has {len(mat)} rows (target words) and {vocab_size} columns (vocab size)")


