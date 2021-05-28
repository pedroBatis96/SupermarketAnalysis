from multiprocessing import Process, Manager

import numpy as np
import pandas as pd
import time

import json

from argparse import ArgumentParser

from helpers.DataMiningHelper import calc_tops_by_nif, fp_growth, calc_random_pick_up_prob
from helpers.DrawHelper import drawMatrix
from helpers.GeneticAlgoritm import TheGenetic
from helpers.ReceiptAnalyser import ReceiptAnalyser, analyse_receipts
from helpers.ExplanationAnalyser import analyse_explanations
from helpers.StatisticsHelper import get_normal_stamina_distribuition, get_tops,draw_stamina_distribuition
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


def start_testing():
    genetic = TheGenetic('sales')
    genetic.start_train()

    genetic = TheGenetic('profit')
    genetic.start_train()


def test_specific(arr):
    arr = 1 + np.array(arr)
    g_obj = GraphHelper()
    graph = g_obj.create_supermarket()
    simulator = TheSimulator(graph)
    simulator.prepare_clients(15000)
    simulator.prepare_products(arr)
    print("test")
    print(simulator.begin_simulation())

def draw():
    drawMatrix()



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('task', choices={'t_s', 'train', 'fpgrowth', 'top100', 'normal','top10','drawstamina','drawmatrix'})
    args = parser.parse_args()
    start = time.time()

    if args.task == 't_s':
        test_specific([160, 53, 96, 39, 79, 17, 146, 79, 160, 41, 93, 45, 2, 80, 95, 78, 72, 106, 100, 59, 29, 50, 18, 76, 75, 155, 104, 98, 99, 3, 56, 63, 33, 158, 14, 23, 15, 157, 111, 8, 132, 121, 42, 1, 2, 52, 44, 2, 54, 121, 136, 6, 67, 115, 70, 112, 110, 35, 22, 3, 43, 14, 5, 39, 1, 121, 43, 4, 0, 64, 0, 56, 89, 32, 159, 134, 18, 0, 81, 74, 3, 48, 44, 46, 51, 2, 1, 1, 37, 36, 36, 47, 69, 37, 161, 97, 83, 82, 162, 0, 79, 3, 17, 94, 126, 154, 108, 84, 79, 17, 146, 17, 153, 147, 102, 124, 90, 63, 91, 146, 83, 62, 68, 82, 83, 87, 150, 3, 77, 84, 148, 66, 61, 0, 155, 109, 28, 4, 17, 73, 66, 127, 2, 64, 88, 158, 80, 49, 1, 107, 118, 131, 143, 65, 7, 34, 31, 34, 7, 129, 10, 123, 122, 55, 40, 68, 82, 105, 25, 130, 159, 24, 61, 60, 59, 60, 33, 163, 138, 11, 120, 19, 86, 92, 137, 141, 13, 27, 149, 38, 101, 85, 125, 87, 6, 6, 9, 135, 133, 103, 35, 119, 144, 152, 71, 86, 117, 87, 18, 24, 31, 123, 14, 7, 8, 58, 116, 142, 113, 15, 114, 16, 12, 20, 21, 154, 26, 128, 139, 156, 140, 164, 130, 32, 32, 79, 132, 146, 151, 151, 23, 8, 63, 4, 145, 128, 57, 30])
    elif args.task == 'train':
        start_testing()
    elif args.task == 'fpgrowth':
        fp_growth()
    elif args.task == 'top100':
        calc_tops_by_nif()
    elif args.task == 'normal':
        get_normal_stamina_distribuition()
    elif args.task == 'top10':
        get_tops()
    elif args.task == 'drawstamina':
        draw_stamina_distribuition()
    elif args.task == 'drawmatrix':
        draw()
    else:
        start_testing()

    print(time.time() - start)
