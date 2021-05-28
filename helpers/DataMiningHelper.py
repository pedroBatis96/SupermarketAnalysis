import json

import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
# mlxtend.frequent_patterns import fpgrowth
from sklearn import preprocessing
from fpgrowth_py import fpgrowth


# para cada recibo de cada pasta, analisa e retira a informação relevante
def fp_growth():
    teste = pd.read_csv('data/products.csv', encoding='utf-8', usecols=["ID", "Nome"], index_col="ID")

    # concatena os csvs de cada pasta
    frames = []
    for i in range(0, 50):
        df = pd.read_csv('data/receipt_{}.csv'.format(i), usecols=['nif', 'total', 'products'])
        frames.append(df)
    result = pd.concat(frames)

    # vai buscar apenas os products, que correspondente a unique products de cada recibo
    dataset = result['products'].to_numpy()

    # transforma a lista guardada em lista python
    for d in range(0, len(dataset)):
        dataset[d] = json.loads(dataset[d])

    for fp in [0.1,0.05,0.01]:
        print(fp)
        freqItemSet, rules = fpgrowth(dataset, minSupRatio=fp, minConf=0.6)
        new_df = pd.DataFrame(rules)
        new_df.to_csv(f'respostas/fpgrowth_{fp}.csv', encoding='utf-8')

    # cria one hot encoding e atribui a um csv
    #te = TransactionEncoder()
    #te_ary = te.fit(dataset).transform(dataset)
    #df = pd.DataFrame(te_ary, columns=te.columns_)
    #df = df.rename(columns=teste.to_dict()['Nome'])

   # for fp in [0.01,0.05,0.1,0.5]:
        #print(fp)
        # usa o one hot encoding para calcular o fpgrowth
        #df_new = fpgrowth(df, min_support=fp, use_colnames=True)
        #df_new.to_csv(f'respostas/fpgrowth_{fp}.csv', encoding='utf-8')


# para cada recibo de cada pasta, analisa e retira a informação relevante
def calc_tops_by_nif():
    # concatena os csvs de cada pasta
    frames = []
    for i in range(0, 50):
        df = pd.read_csv('data/receipt_{}.csv'.format(i), usecols=['nif', 'total', 'products', 'products_all'],
                         dtype={'total': float})
        frames.append(df)
    result = pd.concat(frames)

    # agrupa por nif e faz o sum dos totais
    result = result.groupby(by="nif").sum('total')
    result.reindex()

    # vai buscar apenas os profits do total
    top_profit_nif = pd.read_csv('data/totals/niftotals.csv'.format(i), usecols=['nif', 'total_profit'],
                                 dtype={'total_profit': float})

    # cria um csv que junta os profits com os sales
    all_tops = pd.merge(result, top_profit_nif, on='nif')
    all_tops.to_csv('data/totals/niftotals.csv', encoding='utf-8')

    # cria dois dataframes, um para os totais e outro para os profits
    top_sales_nif = all_tops[['nif', 'total']].groupby(by="nif")
    top_profits_nif = all_tops[['nif', 'total_profit']].groupby(by="nif")

    # vai buscar os top 100 de cada categoria e guarda num csv
    top_sales_nif = top_sales_nif.sum('total').sort_values(ascending=False, by="total").head(100)
    top_profits_nif = top_profits_nif.sum('total_profit').sort_values(ascending=False, by="total_profit").head(100)

    top_sales_nif.to_csv('data/totals/top_sales_nif.csv', encoding='utf-8')
    top_profits_nif.to_csv('data/totals/top_profits_nif.csv', encoding='utf-8')


# conta o número de vezes que um produto aparece e compara com a wish list (ver produtos com prob de serem apanhados)
def calc_random_pick_up_prob(calc=False):
    if calc:
        prod_df = pd.read_csv('data/products.csv', encoding='utf-8', usecols=["ID", "Nome", "Preço", "Margem Lucro"],
                              index_col="ID")

        # concatena todos os recibos
        frames = []
        for i in range(0, 50):
            df = pd.read_csv('data/receipt_{}.csv'.format(i), usecols=['products_all'],
                             dtype={'total': float})
            frames.append(df)
        result_receipts = pd.concat(frames)['products_all'].apply(json.loads).to_numpy()

        # concatena todos as explicações
        frames = []
        for i in range(0, 50):
            df = pd.read_csv('data/explanations/explanation_{}.csv'.format(i), usecols=['products'],
                             dtype={'total': float})
            frames.append(df)
        result_explanations = pd.concat(frames)['products'].apply(json.loads).to_numpy()

        # verifica os produtos de diferença entre recibo e wish list (ver quais foram apanhados sem serem desejados)
        n_new_products = np.zeros(len(prod_df.index), dtype=int)
        for i, item in enumerate(result_receipts):
            item_aux = np.setdiff1d(item, result_explanations[i])
            for j in item_aux:
                n_new_products[j - 1] += 1
        new_df = prod_df.copy()

        # acrescenta a coluna e guarda num csv
        new_df['ProbPickUp'] = n_new_products
        new_df.to_csv('data/totals/ProbPickUp.csv', encoding='utf-8')
    else:
        # le o csv com as probabilidades de serem apanhados random
        new_df = pd.read_csv('data/totals/ProbPickUp.csv', encoding='utf-8',
                             usecols=["ID", "ProbPickUp"],
                             index_col="ID", dtype={"ProbPickUp": int})

    # reshape e normalizados
    normalized_df = preprocessing.normalize(new_df['ProbPickUp'].to_numpy().reshape(1, -1))
    new_df["ProbPickUp"] = normalized_df[0,:]
    return new_df


# calcula o profit dos nifs
def calc_nif_top_profits():
    product_df = pd.read_csv('data/products.csv', encoding='utf-8', usecols=["ID", "Preço", "Margem Lucro"],
                             index_col="ID")

    # concatena todos os recibos
    frames = []
    for i in range(0, 50):
        df = pd.read_csv('data/receipt_{}.csv'.format(i), usecols=['nif', 'total', 'products', 'products_all'],
                         dtype={'total': float})
        frames.append(df)

    result = pd.concat(frames)

    # agrupa por nif
    df_group = result.groupby(by="nif")

    rows = np.empty([len(result), 2], dtype=dict)

    # percorre todos os grupos, e calcula o profit que cada ninf atingiu
    for i, (group_name, df_group) in enumerate(result):
        total = 0
        products_all = df_group['products_all'].apply(json.loads).to_numpy()

        for shelf in products_all:
            for product in shelf:
                product_info = product_df.loc[product].to_numpy(dtype=float)
                total += (float(product_info[0]) * (float(product_info[1]) / 100))
        rows[i][0] = group_name
        rows[i][1] = total

    # guarda o produto final num csv
    product_profit_df = pd.DataFrame(rows, columns=['nif', 'total_profit'])
    product_profit_df.to_csv('data/totals/niftotals.csv', encoding='utf-8')

    # faz print aos top 100
    product_profit_df.reindex()
    print(product_profit_df.sort_values(ascending=False, by="total").head(100))
