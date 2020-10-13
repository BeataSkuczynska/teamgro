import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle5
import tqdm as tqdm
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from pandas import read_csv, read_excel


def load_xlsx(fpath):
    return read_excel(fpath)


def load_csv(fpath, sep=','):
    return read_csv(fpath, sep=sep)


def save_summary(df, fpath, colname, sep=";"):
    """Groups dataframe by specified column, prints stats and saves grouping to CSV file"""
    grouped = df[colname].value_counts()
    print('Total rows: ', grouped.sum())
    print('Rows with multiple items: ', grouped[grouped.index.str.contains(",")].sum())
    grouped.to_csv(fpath, sep=sep)


def create_sentences_dist(series):
    """Create data frame with given series of texts and its length in words.
    For simplification, words are demarked with surrounding spaces."""
    df = pd.DataFrame(series)
    df['length'] = np.zeros(df.shape[0])
    for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
        length = len(row.iloc[0].split(" "))
        df.iloc[idx, 1] = length
    return df


def plot_hist(df, col='length'):
    """Plots histogram of values from dataframe. Specifying column is optional."""
    mean = df[col].mean()
    plt.hist(df[col], bins=150)
    plt.axvline(mean, color='k', linestyle='dashed', linewidth=1)
    plt.xlim(0, 300)
    plt.xticks(range(0, 300, 20))
    plt.show()


def detect_lang(df, col):
    """Detects language of text in specified column in data frame."""
    df['lang'] = ''
    for id, row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
        try:
            lang = detect(row[col])
        except LangDetectException:
            lang = 'unk'
        df.iloc[id, -1] = lang
    return df


# functions for saving/loading the dictionary
def load_obj(path):
    with open(path, 'rb') as f:
        return pickle5.load(f)


def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def extract_similar(df, thresh):
    similar_idxs = []

    for idx, row in tqdm.tqdm(df.iterrows()):
        if row.iloc[1][0][1] > thresh:
            similar_idxs.append(idx)

    return df.iloc[similar_idxs, :]


#### Examples of usage
# eu_path = "data/EUvsDisinfo/eu_vs_disinfo.xlsx"
# eu_df = load_xlsx(eu_path)
# save_summary(eu_df, "data/EUvsDisinfo/lang_summary.csv", 'Language / Targeted Audience')
# out = create_sentences_dist(eu_df['Summary of the Disinformation'])
# out['length'].to_csv("data/EUvsDisinfo/sentences_dist.csv")

# fake_path = 'data/FakeCovid/Dataset II(Tweets)/Labelled_Tweet.csv'
# fake = load_csv(fake_path)
# fake = detect_lang(fake, 'text')
# save_summary(fake, 'data/FakeCovid/Dataset II(Tweets)/labelled_labels_summary.csv', 'tweet_class')
# out = create_sentences_dist(fake['text'])
# out['length'].to_csv("data/FakeCovid/Dataset II(Tweets)/labelled_sentences_dist.csv")

final_dict = load_obj('obj/results_dict.pkl')
results = pd.DataFrame(final_dict.items(), columns=["Tweet", "Top 5 Summaries and Scores"])
similar = extract_similar(results, 0.6)
similar.to_csv('data/FakeCovid/Dataset II(Tweets)/best_results_0.6.csv', sep=';')

print()
