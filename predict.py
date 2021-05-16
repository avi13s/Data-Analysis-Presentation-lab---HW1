import argparse
import numpy as np
import pandas as pd
import pickle as pkl
import xgboost as xgb


# Parsing script arguments
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('tsv_path', type=str, help='tsv file path')
args = parser.parse_args()

# Reading input TSV
data = pd.read_csv(args.tsv_path, sep="\t")


# Prediction code
def get_first_value(s):
    if s.strip("{[]}"):
        return int(s.strip("{[]}").split(", ")[0].split(": ")[1])
    return -1


def get_first(s):
    if s.strip("{[]}"):
        return s.strip("{[]}").split(", ")[0].split(": ")[1]
    return -1


def prepare(dataframe, features, response):
    df = dataframe.copy()
    df["backdrop_path"].fillna("", inplace=True)
    df["belongs_to_collection"].fillna("", inplace=True)
    df["homepage"].fillna("", inplace=True)
    df["poster_path"].fillna("", inplace=True)
    df["tagline"].fillna("", inplace=True)
    df['runtime'].fillna(df["runtime"].mean(), inplace=True)
    df['genre'] = df['genres'].apply(get_first_value)
    df['genre'] = df.genre.astype('category').cat.codes
    df['company'] = df['production_companies'].apply(get_first_value)
    df['company'] = df.genre.astype('category').cat.codes
    df['country'] = df['production_countries'].apply(get_first)
    df['country'] = df.genre.astype('category').cat.codes
    df['spoken_language'] = df['spoken_languages'].apply(get_first)
    df['spoken_language'] = df.genre.astype('category').cat.codes
    df['language'] = df.original_language.astype('category').cat.codes
    return df[features].to_numpy(), df[response].to_numpy()


feats = ['budget', 'popularity', 'runtime', 'vote_average', 'vote_count',
            'genre', 'company', 'country', 'spoken_language', 'language']
resp = 'revenue'
X_pred, _ = prepare(data, feats, resp)
with open('model.pkl', 'rb') as f:
    predictor = pkl.load(f)
y_hat = predictor.predict(X_pred)

# Example:
prediction_df = pd.DataFrame(columns=['id', 'revenue'])
prediction_df['id'] = data['id']
prediction_df['revenue'] = y_hat
####

# TODO - How to export prediction results
prediction_df.to_csv("prediction.csv", index=False, header=False)


# ### Utility function to calculate RMSLE
# def rmsle(y_true, y_pred):
#     """
#     Calculates Root Mean Squared Logarithmic Error between two input vectors
#     :param y_true: 1-d array, ground truth vector
#     :param y_pred: 1-d array, prediction vector
#     :return: float, RMSLE score between two input vectors
#     """
#     assert y_true.shape == y_pred.shape, \
#         ValueError("Mismatched dimensions between input vectors: {}, {}".format(y_true.shape, y_pred.shape))
#     return np.sqrt((1/len(y_true)) * np.sum(np.power(np.log(y_true + 1) - np.log(y_pred + 1), 2)))
#
#
# ### Example - Calculating RMSLE
# res = rmsle(data['revenue'], prediction_df['revenue'])
# print("RMSLE is: {:.6f}".format(res))


