from multiprocessing import Process, Manager

import numpy as np
import pandas as pd
import time

from argparse import ArgumentParser

from helpers.DataMiningHelper import calc_tops_by_nif, fp_growth,calc_random_pick_up_prob
from helpers.ReceiptAnalyser import ReceiptAnalyser, analyse_receipts
from helpers.ExplanationAnalyser import analyse_explanations
from helpers.StatisticsHelper import get_normal_stamina_distribuition
from helpers.TheSimulator import TheSimulator

from helpers.GraphHelper import GraphHelper
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
    product_dataframe = pd.read_csv('../Produtos.txt', delimiter='\t', encoding="utf-8")
    product_dataframe.index += 1
    probability = calc_probability(product_dataframe)

    product_dataframe['Nome'] = product_dataframe['Nome'].str.strip()
    product_dataframe['prob'] = probability
    product_dataframe.index.name = 'ID'
    product_dataframe.to_csv('data/products.csv', encoding="utf-8")


# para cada recibo de cada pasta, analisa e retira a informação relevante
def start_receipt_analysis():
    try:
        manager = Manager()
        return_dict = manager.dict()
        open('data/receipt_total.csv', 'w').close()

        threads = [
            Process(target=analyse_receipts, args=(0, 10, return_dict)),
            Process(target=analyse_receipts, args=(10, 20, return_dict)),
            Process(target=analyse_receipts, args=(20, 30, return_dict)),
            Process(target=analyse_receipts, args=(30, 40, return_dict)),
            Process(target=analyse_receipts, args=(40, 50, return_dict))
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        product_dataframe = pd.read_csv('data/products.csv', encoding="utf-8", usecols=['ID', 'Nome'], index_col=["ID"])
        r_t = pd.DataFrame.from_dict(return_dict, orient='index')
        totals = []
        for i in range(0, 165):
            totals.append(r_t[i].sum())
        product_dataframe['TotalSales'] = totals
        product_dataframe.to_csv('data/receipt_total.csv')

    except Exception as e:
        print(e)


# para cada recibo de cada pasta, analisa e retira a informação relevante
def start_explanation_analysis():
    try:
        threads = [
            Process(target=analyse_explanations, args=(0, 10)),
            Process(target=analyse_explanations, args=(10, 20)),
            Process(target=analyse_explanations, args=(20, 30)),
            Process(target=analyse_explanations, args=(30, 40)),
            Process(target=analyse_explanations, args=(40, 50))
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

    except Exception as e:
        print(e)


def create_graph():
    g_obj = GraphHelper()
    graph = g_obj.create_supermarket()
    simulator = TheSimulator(graph)



if __name__ == '__main__':
    start = time.time()

    # calc_random_pick_up_prob()
    create_graph()

    # criar produto
    # create_products()

    # analise de recibos e explicações
    # start_receipt_analysis()
    # start_explanation_analysis()

    # teste
    # analyse_receipts(0, 1,{})
    # analyse_explanations(0, 1)

    # probabilidades
    # fp_growth()
    # calc_p_statistics()
    # get_tops()

    # get_normal_stamina_distribuition()
    # calc_tops_by_nif()

    print(time.time() - start)
