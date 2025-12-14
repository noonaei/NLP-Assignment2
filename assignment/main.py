import json
import time
import embeddings
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from scipy import stats


VOCABULARY_SIZE = 10000
WINDOW_SIZE = 0  # todo: find the window size that works best for you


def predict_words_valence(train_file: str, test_file: str, data_path: str, is_dense_embedding: bool) -> (float, float):
    print(f'starting regression with {train_file}, evaluating on {test_file}, '
          f'and dense word embedding {is_dense_embedding}')

    if is_dense_embedding:
        X_train, y_train, X_test, y_test = embeddings.build_dense_matrices(train_file, test_file)



    print(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}')
    print(f'X_test shape: {X_test.shape}, y_test shape: {y_test.shape}')

    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)

    model = sm.OLS(y_train, X_train)
    result = model.fit()

    print(result.summary())

    y_pred = result.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    corr = stats.pearsonr(y_test, y_pred)[0]
    #returns (correlation_coefficient, p_value)


    return mse , corr  #mse, corr


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
