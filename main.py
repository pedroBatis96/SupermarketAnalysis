import csv
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
from helpers.StatisticsHelper import get_normal_stamina_distribuition, get_tops, draw_stamina_distribuition
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


def order_fp():
    df = pd.read_csv('respostas/fpgrowth_0.5.csv', index_col=0)
    df.sort_values(by='2', ascending=False).to_csv('respostas/fpgrowth_0.5.csv')


def write_best():
    test = [18, 133, 1, 4, 4, 82, 89, 74, 21, 39, 67, 76, 1, 7, 103, 85, 3, 14,
            92, 9, 110, 96, 1, 4, 31, 114, 36, 26, 150, 5, 104, 72, 18, 66, 81, 117,
            3, 80, 19, 88, 115, 17, 13, 30, 163, 153, 80, 15, 16, 2, 85, 113, 60, 53,
            18, 50, 125, 86, 1, 98, 62, 165, 64, 62, 45, 73, 156, 142, 154, 138, 63, 56,
            77, 4, 97, 129, 65, 64, 65, 69, 8, 152, 87, 122, 61, 58, 40, 6, 51, 106,
            109, 119, 49, 67, 42, 69, 84, 61, 91, 25, 2, 18, 129, 87, 118, 155, 33, 57,
            155, 29, 101, 128, 100, 79, 28, 88, 27, 59, 8, 22, 83, 1, 71, 107, 112, 141,
            140, 3, 16, 131, 121, 48, 32, 33, 47, 124, 34, 10, 54, 75, 38, 120, 149, 41,
            133, 2, 84, 88, 43, 60, 122, 123, 68, 8, 9, 24, 147, 111, 124, 90, 160, 81,
            64, 34, 55, 18, 33, 159, 12, 40, 2, 83, 131, 134, 84, 3, 37, 19, 146, 44,
            147, 158, 160, 7, 5, 25, 78, 127, 145, 105, 162, 38, 23, 159, 95, 15, 52, 2,
            37, 20, 45, 137, 144, 9, 44, 147, 70, 164, 157, 143, 116, 35, 24, 7, 36, 32,
            161, 35, 136, 19, 94, 147, 4, 161, 80, 57, 11, 5, 151, 102, 108, 122, 156, 132,
            3, 126, 152, 139, 83, 93, 80, 80, 99, 15, 46, 130, 135, 148]

    #pd.DataFrame(test,type=int).to_csv("respostas/top_value.csv", header=None, index=None)
    with open("respostas/top_value.csv", 'w', newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(np.array(test,dtype=int))

    test2 = [123, 143, 1, 3, 3, 88, 27, 2, 63, 88, 108, 1, 111, 154, 80, 94, 83, 11, 60, 7, 44, 122, 120, 38, 119, 20,
             114, 31, 124, 40, 45, 53, 141, 25, 1, 156, 106, 2, 69, 102, 39, 67, 118, 113, 73, 5, 77, 104, 3, 101, 70,
             87, 48, 33, 116, 47, 159, 83, 126, 148, 12, 9, 19, 18, 84, 122, 151, 7, 32, 18, 149, 58, 121, 142, 90, 37,
             68, 18, 2, 50, 5, 19, 133, 91, 56, 107, 146, 22, 61, 109, 6, 13, 129, 150, 130, 36, 24, 9, 35, 147, 163,
             137, 4, 15, 112, 16, 35, 110, 16, 60, 165, 33, 152, 135, 100, 34, 147, 84, 55, 133, 42, 52, 8, 158, 14, 7,
             32, 115, 28, 5, 34, 54, 97, 99, 64, 57, 66, 40, 80, 45, 51, 3, 162, 26, 161, 92, 4, 153, 9, 49, 105, 80,
             152, 160, 138, 96, 18, 136, 62, 81, 62, 88, 117, 65, 85, 80, 71, 95, 36, 1, 140, 128, 46, 80, 74, 157, 81,
             24, 33, 15, 131, 147, 4, 89, 161, 127, 59, 1, 2, 147, 155, 139, 129, 124, 2, 87, 23, 78, 164, 144, 17, 156,
             4, 3, 79, 8, 85, 64, 75, 21, 159, 98, 15, 160, 86, 76, 103, 122, 134, 37, 69, 8, 30, 132, 93, 125, 145, 57,
             65, 4, 38, 44, 131, 10, 155, 29, 25, 19, 43, 84, 83, 64, 18, 67, 82, 72, 41, 61]

    with open("respostas/top_profit.csv", 'w', newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(np.array(test2,dtype=int))

    #pd.DataFrame(test2,type=int).to_csv("respostas/top_profit.csv", header=None, index=None)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('task',
                        choices={'t_s', 'train', 'fpgrowth', 'top100', 'normal', 'top10', 'drawstamina', 'drawmatrix',
                                 'orderfp'})
    args = parser.parse_args()
    start = time.time()

    if args.task == 'train':
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
    elif args.task == 'orderfp':
        write_best()
    else:
        start_testing()

    print(time.time() - start)
