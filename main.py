import numpy as np
import pandas as pd
from helpers.ReceiptAnalyser import ReceiptAnalyser
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))


# calcula as probabilidades (através do apriori) para cada produto, para adicionar ao csv
def calc_probability(data):
    probability = np.array([])
    for index, row in data.iterrows():
        if index == 1:
            p_value = row[-1]
        else:
            p_value = row[-1] - (data.loc[(index - 1), :][-1])

        # adiciona as probabilidades ao array
        probability = np.append(probability, [p_value])
    return probability


# cria os produtos como csv com a coluna prob
def create_products():
    product_dataframe = pd.read_csv('../Produtos.csv')
    product_dataframe.index += 1
    probability = calc_probability(product_dataframe)
    product_dataframe['prob'] = probability
    product_dataframe.to_csv('data/products.csv')


# para cada recibo de cada pasta, analisa e retira a informação relevante
def analyse_receipts():
    ra = ReceiptAnalyser()
    f = open("../receipts/0/receipt_0.txt", "r", encoding="utf-8")
    s = f.read()
    ra.analyze_receipt(s)


if __name__ == '__main__':
    # create_products()
    analyse_receipts()
