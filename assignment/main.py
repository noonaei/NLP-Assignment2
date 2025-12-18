import json
import time
import dense
import long_and_sparse as lsp
import utils as utls
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error
import numpy as np

VOCABULARY_SIZE = 10000
WINDOW_SIZE = 3
VOCAB_PICKLE_FILE = './pkl/vocab.pkl'
MAX_LINES = 7_000_000


def predict_words_valence(train_file: str, test_file: str, data_path: str, is_dense_embedding: bool) -> (float, float):
    print(f'starting regression with {train_file}, evaluating on {test_file}, '
          f'and dense word embedding {is_dense_embedding}')

    if is_dense_embedding:
        X_train, y_train, X_test, y_test = embeddings.build_dense_matrices(train_file, test_file)
    else:
        # Load pre-computed from 7M lines vocab from pickle
        vocab = utls.load_pickle(VOCAB_PICKLE_FILE)

        # Load training data and separate words from valence scores
        train_target_words, y_train = utls.read_csv_to_tuples(train_file)
        test_target_words, y_test = utls.read_csv_to_tuples(test_file)

        # Combine train and test words to build single co-occurrence matrix
        data_for_co_occurrence_mat = train_target_words + test_target_words

        # compute the co-occurrence matrix
        co_occurrence_mat = lsp.build_co_occurrence_matrix(data_path, data_for_co_occurrence_mat, vocab, WINDOW_SIZE, MAX_LINES)

        # normalize the co-occurrence matrix
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

    return mse , corr


if __name__ == '__main__':
    start_time = time.time()
    with open('config.json', 'r') as json_file:
        config = json.load(json_file)

    mse, corr = predict_words_valence(
        config['train'],
        config['test'],
        config['wiki_data'],
        config["word_embedding_dense"])

    elapsed_time = time.time() - start_time
    print(f"elapsed time: {elapsed_time: .2f} seconds")

    print(f'test set evaluation results: MSE: {mse: .3f}, '
          f'Pearsons correlation: {corr: .3f}')
