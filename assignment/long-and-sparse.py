"""Word valence prediction using co-occurrence matrix representations.

Builds word embeddings from co-occurrence statistics and trains a linear regression
model to predict valence scores for words based on their distributional context.
"""

from utils_michael import *
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error
import numpy as np


MAX_LINES = 7_000_000  # 7 million lines
TOP_K = 10000
CORPUS_FILE_NAME = './data/en.wikipedia2018.10M.txt'
VOCAB_PICKLE_FILE = './vocab.pkl'
WINDOW_SIZE = 3


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


# Load pre-computed from 7M lines vocab from pickle
vocab = load_pickle(VOCAB_PICKLE_FILE)

# Load training data and separate words from valence scores
train_target_words, y_train = read_csv_to_tuples('./data/nrc-valence-scores.trn.csv')
test_target_words, y_test = read_csv_to_tuples('./data/nrc-valence-scores.tst.csv')

# Combine train and test words to build single co-occurrence matrix
data_for_co_occurrence_mat = train_target_words + test_target_words

co_occurrence_mat = build_co_occurrence_matrix(CORPUS_FILE_NAME, data_for_co_occurrence_mat, vocab, WINDOW_SIZE, MAX_LINES)

# L2 normalize each row to create unit-length word vectors
log1p_corr_mat = np.log1p(co_occurrence_mat)
mat_final = normalize(log1p_corr_mat, norm='l2', axis=1)

# Split normalized matrix back into train and test sets
X_train = mat_final[:len(train_target_words), :]
X_test = mat_final[-len(test_target_words):, :]

reg = LinearRegression().fit(X_train, y_train)
pred_test = reg.predict(X_test)


corr_mat = np.corrcoef(pred_test, y_test)
corr = corr_mat[0, 1]
mse = mean_squared_error(y_test, pred_test)


