import json

import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from ast import literal_eval


# para cada recibo de cada pasta, analisa e retira a informação relevante
def fp_growth():
    teste = pd.read_csv('data/products.csv', encoding='utf-8', usecols=["ID", "Nome"], index_col="ID")


    frames = []
    for i in range(0, 86):
        df = pd.read_csv('data/receipt_{}.csv'.format(i), usecols=['products'])
        frames.append(df)

    result = pd.concat(frames)
    dataset = result['products'].to_numpy()

    for d in range(0, len(dataset)):
        dataset[d] = json.loads(dataset[d])

    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    df = df.rename(columns=teste.to_dict()['Nome'])
    df_new = fpgrowth(df, min_support=0.3, use_colnames=True)
    df_new.to_csv('data/fpgrowth.csv', encoding='utf-8')
