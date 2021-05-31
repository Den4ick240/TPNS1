import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from get_dataset import get_dataset


def dist_properties(data):
    columns = data.columns
    d = {}
    d['Кол-во'] = []
    d['% пропусков'] = []
    d['Минимум'] = []
    d['Максимум'] = []
    d['Среднее'] = []
    d['Мощность'] = []
    d['% уникальных'] = []
    d['Первый квартиль(0.25)'] = []
    d['Медиана'] = []
    d['Второй квартиль(0.75)'] = []
    d['Стандартное отклонение'] = []
    for h in data.columns:
        d['Кол-во'].append(data[h].count())
        d['% пропусков'].append(data[h].isna().sum() / len(data) * 100)
        d['Минимум'].append(data[h].min())
        d['Максимум'].append(data[h].max())
        d['Среднее'].append(data[h].mean())
        d['Мощность'].append(data[h].nunique())
        d['% уникальных'].append(data[h].nunique() / data[h].count() * 100)
        d['Первый квартиль(0.25)'].append(data[h].quantile(0.25))
        d['Медиана'].append(data[h].median())
        d['Второй квартиль(0.75)'].append(data[h].quantile(0.75))
        d['Стандартное отклонение'].append(data[h].std())
    return pd.DataFrame(d, columns)


def remove_skipped(data, tab):
    removed = []
    cat_index = []
    cont_index = []
    for i in tab.index:
        if tab['% пропусков'][i] > 60 and i != 'G_total':
            print(i, ' больше 60% пропусков')
            removed.append(i)
            continue
        if tab['Мощность'][i] == 1:
            print(i, ' мощность 1')
            removed.append(i)
            continue
        if tab['Мощность'][i] < 25:
            cat_index.append(i)
        else:
            cont_index.append(i)
    data.drop(removed, axis=1, inplace=True)
    return cont_index, cat_index


def cat_dist_properties(data, cat_index):
    d = {'Кол-во': [], '% пропусков': [], 'Мощность': []}
    for j in (0, 1):
        d['Мода' + str(j + 1)] = []
        d['Частота моды' + str(j + 1)] = []
        d['% моды' + str(j + 1)] = []

    for i in cat_index:
        d['Кол-во'].append(data[i].count())
        d['% пропусков'].append(data[i].isna().sum() / len(data) * 100)
        d['Мощность'].append(data[i].nunique())
        vc = data[i].value_counts()
        for j in (0, 1):
            m = vc.index[j]
            m_count = vc[m]
            m_p = m_count / d['Кол-во'][cat_index.index(i)] * 100
            d['Мода' + str(j + 1)].append(m)
            d['Частота моды' + str(j + 1)].append(m_count)
            d['% моды' + str(j + 1)].append(m_p)
    return pd.DataFrame(d, cat_index)


def remove_emissions(data, cont_index, tab):
    normal_dist = ['Руст', 'Рзаб', 'Рлин', 'Рлин_2', 'Дебит кон нестабильный']
    for i in cont_index:
        if i in normal_dist:
            bot = tab['Среднее'][i] - 2 * tab['Стандартное отклонение'][i]
            top = tab['Среднее'][i] + 2 * tab['Стандартное отклонение'][i]
        else:
            x025 = tab['Первый квартиль(0.25)'][i]
            x075 = tab['Второй квартиль(0.75)'][i]
            bot = x025 - 1.5 * (x075 - x025)
            top = x075 + 1.5 * (x075 - x025)
        print(i, bot, top)
        for j, row in data.iterrows():
            if data[i][j] < bot or data[i][j] > top:
                print(j)
                if i == 'КГФ':
                    data.drop(index=j, inplace=True)
                else:
                    data[i][j] = float('nan')
                    if tab['% пропусков'][i] < 30:
                        data[i][j] = tab['Медиана'][i]

    return data.reset_index(drop=True)


def get_gain_ratio(data):
    N = data.shape[0]
    n = int(np.log2(N)) + 1
    ct = pd.DataFrame(index=data.index, columns=data.columns)
    for column in ct:
        min = data[column].min()
        max = data[column].max()
        step = (max - min) / n
        for i in range(N):
            if not np.isnan(data[column][i]):
                interval = int((data[column][i] - min) / step)
                if interval == n:
                    interval -= 1
                ct[column][i] = interval
            else:
                ct[column][i] = -1
    print(ct)
    ct.astype('int32')
    freq_T = np.zeros((n + 1, n), dtype=int)
    for i in range(N):
        freq_T[ct['G_total'][i] + 1, ct['КГФ'][i]] += 1
    print(freq_T)

    info_T = 0
    for i in range(n + 1):
        for j in range(n):
            ft = freq_T[i, j]
            if ft != 0:
                info_T -= (ft / N) * np.log2(ft / N)
    print(info_T)
    gain_ratio = {}
    for column in ct.columns:
        if column != 'КГФ' and column != 'G_total':

            info_x_T = 0  # Оценка количеств информации после разбиения множества T по column
            split_info_x = 0
            for i in range(n):  # проходимся по классам разбиения
                Ni = 0
                freq_x_T = np.zeros_like(freq_T)  # Для каждого класса разбиения из column - мощности классов Cj
                for j in range(N):
                    x = ct[column][j]
                    if x == i:
                        Ni += 1
                        freq_x_T[ct['G_total'][j] + 1, ct['КГФ'][j]] += 1
                info_Ti = 0  # Оценка кол-ва информации для определения класса из Ti
                if Ni != 0:
                    for i in range(n + 1):
                        for j in range(n):
                            if freq_x_T[i, j] != 0:
                                info_Ti -= (freq_x_T[i, j] / Ni) * np.log2(freq_x_T[i, j] / Ni)
                    info_x_T += (Ni / N) * info_Ti
                    split_info_x -= (Ni / N) * np.log2((Ni / N))
            gain_ratio[column] = (info_T - info_x_T) / split_info_x

    vals = list(gain_ratio.values())
    length = len(vals)
    keys = list(gain_ratio.keys())
    fig, ax = plt.subplots(figsize=(7, 7))
    plt.barh(keys, vals, align='center', color='green')
    for i in range(length):
        plt.annotate("%.2f" % vals[i], xy=(vals[i], keys[i]), va='center')
    plt.show()
    return gain_ratio


def correlation_matrix(df):
    corr_matrix = df.corr().to_numpy()
    fig, ax = plt.subplots(figsize=(15, 15))
    im = ax.imshow(corr_matrix)
    ax.xaxis.set(ticks=np.arange(len(df.columns)), ticklabels=df.columns)
    ax.yaxis.set(ticks=np.arange(len(df.columns)), ticklabels=df.columns)
    ax.xaxis.set_tick_params(rotation=90)
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            ax.text(j, i, '{:.2f}'.format(corr_matrix[i, j]), ha='center', va='center',
                    color='w')
    plt.show()
    return corr_matrix


def drop(df, corr_matrix, gain_ratio):
    dropped = []
    for i in range(len(df.columns)):
        col1 = df.columns[i]
        if col1 != 'КГФ' and col1 != 'G_total':
            for j in range(i):
                col2 = df.columns[j]
                if col2 in dropped:
                    continue
                if col2 != 'КГФ' and col2 != 'G_total':
                    if corr_matrix[i, j] > 0.9:
                        drop_f = True
                        for k in range(len(df.columns)):
                            col3 = df.columns[k]
                            if col3 in dropped:
                                continue
                            if abs(corr_matrix[i, k] - corr_matrix[j, k]) > 0.25:
                                drop_f = False
                        if drop_f:
                            print(corr_matrix[i, j], col1, gain_ratio[col1], ' - ', col2, gain_ratio[col2])
                            if gain_ratio[col1] > gain_ratio[col2]:
                                dropped.append(col2)
                            else:
                                dropped.append(col1)
    print('dropped\n', dropped)
    df.drop(columns=dropped, inplace=True)


def main():
    data = get_dataset()

    tab_dist = dist_properties(data)
    cont_index, cat_index = remove_skipped(data, tab_dist)
    tab_cat = cat_dist_properties(data, cat_index)
    print(tab_cat)
    data.rename(columns={cat_index[i]: cat_index[i] + '_категориальный' for i in range(len(cat_index))})
    data.hist(bins=50, figsize=(20, 20), color='green')
    plt.show()
    data = remove_emissions(data, cont_index, tab_dist)
    data.hist(bins=50, figsize=(20, 20), color='green')
    plt.show()
    gain_ratio = get_gain_ratio(data)
    corr_matrix = correlation_matrix(data)
    drop(data, corr_matrix, gain_ratio)


if __name__ == '__main__':
    main()
