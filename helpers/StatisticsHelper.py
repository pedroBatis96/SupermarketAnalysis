import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import seaborn as sns
from scipy.stats import norm


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


def get_normal_stamina_distribuition():
    frames = []

    for i in range(0, 50):
        df = pd.read_csv('data/explanations/explanation_{}.csv'.format(i), usecols=['stamina'],
                         dtype={"stamina": float})
        frames.append(df)

    fig, ax = plt.subplots(1, 1)

    result = pd.concat(frames)
    x = result['stamina'].to_numpy()

    # Stamina histogram
    sns.distplot(x, hist=True, kde=False, color='blue', hist_kws={'edgecolor': 'black'})
    plt.savefig("staminaHistogram.jpg")
    plt.clf()

    # seaborn histogram
    sns.distplot(x, hist=True, kde=True, color='blue', hist_kws={'edgecolor': 'black'},kde_kws={'linewidth': 2})
    plt.savefig("staminaHistogramDensity.jpg")
    plt.clf()

    # Isolate the density
    sns.kdeplot(data=x)
    plt.savefig("staminaDensity.jpg")

    x = x.reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(x)

    plt.show()
