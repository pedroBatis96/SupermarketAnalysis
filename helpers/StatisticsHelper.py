import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth


# para cada recibo de cada pasta, analisa e retira a informação relevante
def calc_p_statistics():
    df_p = pd.read_csv('data/products.csv', encoding='utf-8', index_col="ID")
    df_t = pd.read_csv('data/receipt_total.csv', encoding='utf-8', index_col="ID")

    df_t['TotalSalesValue'] = df_p['Preço'] * df_t['TotalSales']
    df_t['TotalSalesProfit'] = (df_p['Preço'] * (df_p['Margem Lucro'] / 100)) * df_t['TotalSales']

    df_t.to_csv('data/receipt_total.csv', encoding='utf-8')


def get_tops():
    df_t = pd.read_csv('data/receipt_total.csv', encoding='utf-8', index_col="ID")
    topSales = df_t.sort_values(ascending=False, by="TotalSales").head(10)
    topMoney = df_t.sort_values(ascending=False, by="TotalSalesValue").head(10)
    topProfit = df_t.sort_values(ascending=False, by="TotalSalesProfit").head(10)

    print(topSales.head(10))
    print(topMoney.head(10))
    print(topProfit.head(10))
    teste = 1
