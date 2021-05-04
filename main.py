import numpy as np
import pandas as pd
import time

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
    product_dataframe = pd.read_csv('../Produtos.csv', encoding="utf-8")
    product_dataframe.index += 1
    probability = calc_probability(product_dataframe)

    product_dataframe['Nome'] = product_dataframe['Nome'].str.strip()
    product_dataframe['prob'] = probability
    product_dataframe.to_csv('data/products.csv', encoding="utf-8")


# para cada recibo de cada pasta, analisa e retira a informação relevante
def analyse_receipts():
    ra = ReceiptAnalyser()
    for d in range(0, 86):
        receipt = pd.DataFrame(columns=['rindex', 'nif', 'produtos_all', 'produtos', 'grupos', 'total'])

        dir_str = "../receipts/{}/".format(d)
        print("receipts_{}".format(d) + " started:")

        directory = os.fsencode(dir_str)
        i = 0
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".txt"):
                filepath = os.path.join(dir_str, filename)
                f = open(filepath, "r", encoding="utf-8")
                s = f.read()
                information = ra.analyze_receipt(s)
                information['rindex'] = i
                receipt = receipt.append(information, ignore_index=True)
                i += 1
                continue
            else:
                continue
        receipt.set_index('rindex')
        receipt.to_csv('data/receipt_{}.csv'.format(d), encoding="utf-8")

        print("receipts_{}".format(d) + " ended")


if __name__ == '__main__':
    start = time.time()
    # create_products()
    analyse_receipts()
    print(time.time() - start)
