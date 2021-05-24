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
        df = pd.read_csv('data/receipt_{}.csv'.format(i), usecols=['nif', 'total', 'products', 'products_all'],
                         dtype={'total': float})
        frames.append(df)

    # concatenar os resultados
    result = pd.concat(frames)
    result = result.groupby(by="nif").sum('total')
    result.reindex()

    top_profit_nif = pd.read_csv('data/totals/niftotals.csv'.format(i), usecols=['nif', 'total_profit'],
                                 dtype={'total_profit': float})

    all_tops = pd.merge(result, top_profit_nif, on='nif')
    all_tops.to_csv('data/totals/niftotals.csv', encoding='utf-8')

    top_sales_nif = all_tops[['nif', 'total']].groupby(by="nif")
    top_profits_nif = all_tops[['nif', 'total_profit']].groupby(by="nif")

    top_sales_nif = top_sales_nif.sum('total').sort_values(ascending=False, by="total").head(100)
    top_profits_nif = top_profits_nif.sum('total_profit').sort_values(ascending=False, by="total_profit").head(100)

    top_sales_nif.to_csv('data/totals/top_sales_nif.csv', encoding='utf-8')
    top_profits_nif.to_csv('data/totals/top_profits_nif.csv', encoding='utf-8')

    # product_profit_df = pd.DataFrame(columns=['nif', 'total_profit'], dtype={'nif': 'string', 'total_profit': 'float'})
    # = np.empty([len(result),2],dtype=dict)
    # i = 0
    # for i , (group_name, df_group) in enumerate(result):
    #    total = 0
    #    products_all = df_group['products_all'].apply(json.loads).to_numpy()

    #    for shelf in products_all:
    #        for product in shelf:
    #            product_info = product_df.loc[product].to_numpy(dtype=float)
    #            total += (float(product_info[0]) * (float(product_info[1]) / 100))
    #    rows[i][0] = group_name
    #    rows[i][1] = total

    # product_profit_df = pd.DataFrame(rows, columns=['nif', 'total_profit'])
    # product_profit_df.to_csv('data/totals/niftotals.csv', encoding='utf-8')

    # product_profit_df.reindex()
    # print(product_profit_df.sort_values(ascending=False, by="total").head(100))


def count_all(calc=False):
    if calc:
        prod_df = pd.read_csv('data/products.csv', encoding='utf-8', usecols=["ID", "Nome", "Preço", "Margem Lucro"],
                              index_col="ID")

        frames = []
        for i in range(0, 50):
            df = pd.read_csv('data/receipt_{}.csv'.format(i), usecols=['products_all'],
                             dtype={'total': float})
            frames.append(df)

        result_receipts = pd.concat(frames)['products_all'].apply(json.loads).to_numpy()

        frames = []
        for i in range(0, 50):
            df = pd.read_csv('data/explanations/explanation_{}.csv'.format(i), usecols=['products'],
                             dtype={'total': float})
            frames.append(df)

        result_explanations = pd.concat(frames)['products'].apply(json.loads).to_numpy()

        n_new_products = np.zeros(len(prod_df.index), dtype=int)
        for i, item in enumerate(result_receipts):
            item_aux = np.setdiff1d(item, result_explanations[i])
            for j in item_aux:
                n_new_products[j - 1] += 1

        new_df = prod_df.copy()
        new_df['ProbPickUp'] = n_new_products
        new_df.to_csv('data/totals/ProbPickUp.csv', encoding='utf-8')
    else:
        new_df = pd.read_csv('data/totals/ProbPickUp.csv', encoding='utf-8',
                             usecols=["ID", "Nome", "Preço", "Margem Lucro", "ProbPickUp"],
                             index_col="ID")

    print(new_df.sort_values(by='ProbPickUp', ascending=False).head(100))

    # for res in result_receipts:
    #    for item in res:
    #        n_products[item - 1] += 1
