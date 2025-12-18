import csv
from typing import Iterable, List, Tuple, Union
import numpy as np
import gensim.downloader as api


def load_word_valence_csv(path: str) -> List[Tuple[str, float]]:
    """
    Expected CSV format: word,valence
    May contain a header row. Extra columns are ignored.
    """
    rows: List[Tuple[str, float]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:

            if not row:
                continue

            word = row[0].strip()

            # Skip rows with missing valence
            if len(row) < 2:
                continue

            valence_raw = row[1].strip()

            try:
                valence = float(valence_raw)
            except ValueError:
                # If it's not a number (e.g., header or malformed line), skip it
                print(f"could not convert valence at row ${row} Skipping row.")
                continue

            if word:
                rows.append((word, valence))

    return rows


def build_dense_matrices(
    train_data: Union[ str, Iterable[Tuple[str, float]]],
    test_data: Union[str, Iterable[Tuple[str, float]]],
):
    wv = api.load("word2vec-google-news-300")

    # Allow either passing file paths or already-parsed (word, valence) pairs
    if isinstance(train_data, str):
        train_pairs = load_word_valence_csv(train_data)
    else:
        train_pairs = list(train_data)

    if isinstance(test_data, str):
        test_pairs = load_word_valence_csv(test_data)
    else:
        test_pairs = list(test_data)

    X_train: List[np.ndarray] = []
    y_train: List[float] = []
    X_test: List[np.ndarray] = []
    y_test: List[float] = []

    #checks if a word exists in the Word2Vec vocabulary
    in_vocab = wv.key_to_index.__contains__

    for word, valence in train_pairs:
        if in_vocab(word):
            X_train.append(wv[word])
            y_train.append(valence)
        else:
            print(f"{word} not found in vocabulary")

    for word, valence in test_pairs:
        if in_vocab(word):
            X_test.append(wv[word])
            y_test.append(valence)
        else:
            print(f"{word} not found in vocabulary")

    # Convert to numpy arrays so callers can use .shape and ML libs can consume them
    X_train_arr = np.asarray(X_train, dtype=np.float32)
    y_train_arr = np.asarray(y_train, dtype=np.float32)
    X_test_arr = np.asarray(X_test, dtype=np.float32)
    y_test_arr = np.asarray(y_test, dtype=np.float32)

    return X_train_arr, y_train_arr, X_test_arr, y_test_arr
