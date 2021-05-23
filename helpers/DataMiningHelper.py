import json

import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth


# para cada recibo de cada pasta, analisa e retira a informação relevante
def fp_growth():
    teste = pd.read_csv('data/products.csv', encoding='utf-8', usecols=["ID", "Nome"], index_col="ID")

    frames = []
    for i in range(0, 50):
        df = pd.read_csv('data/receipt_{}.csv'.format(i), usecols=['nif', 'total', 'products'])
        frames.append(df)

    result = pd.concat(frames)

    # result.groupby(by="nif").sum('total').max()
    dataset = result['products'].to_numpy()

    for d in range(0, len(dataset)):
        dataset[d] = json.loads(dataset[d])

    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    df = df.rename(columns=teste.to_dict()['Nome'])
    df_new = fpgrowth(df, min_support=0.3, use_colnames=True)
    df_new.to_csv('data/fpgrowth.csv', encoding='utf-8')


# para cada recibo de cada pasta, analisa e retira a informação relevante
def best_client():
    product_df = pd.read_csv('data/products.csv', encoding='utf-8', usecols=["ID", "Preço", "Margem Lucro"],
                             index_col="ID")

    frames = []
    for i in range(0, 50):
        df = pd.read_csv('data/receipt_{}.csv'.format(i), usecols=['nif', 'total', 'products', 'products_all'])
        frames.append(df)

    result = pd.concat(frames)

    top_sales_nif = result.groupby(by="nif").sum('total').sort_values(ascending=False, by="total").head(100)
    print(top_sales_nif)

    result = result.groupby(by="nif")

    #product_profit_df = pd.DataFrame(columns=['nif', 'total_profit'], dtype={'nif': 'string', 'total_profit': 'float'})
    rows = np.empty([len(result),2],dtype=dict)
    i = 0
    for i , (group_name, df_group) in enumerate(result):
        total = 0
        products_all = df_group['products_all'].apply(json.loads).to_numpy()

        for shelf in products_all:
            for product in shelf:
                product_info = product_df.loc[product].to_numpy(dtype=float)
                total += (float(product_info[0]) * (float(product_info[1]) / 100))
        rows[i][0] = group_name
        rows[i][1] = total

    product_profit_df = pd.DataFrame(rows, columns=['nif', 'total_profit'])
    product_profit_df.to_csv('data/totals/niftotals.csv', encoding='utf-8')

    #product_profit_df.reindex()
    #print(product_profit_df.sort_values(ascending=False, by="total").head(100))
