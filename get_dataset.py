import pandas as pd
import chardet

def get_dataset():
    with open('data/ID_data_mass_18122012.csv', 'rb') as f:
        result = chardet.detect(f.read())
        print(result)
    data = pd.read_csv('data/ID_data_mass_18122012.csv', sep=';', encoding=result['encoding'])
    data = data.apply(lambda x: x.str.replace(',', '.'))
    headers = data.values[0, 2:].tolist()
    headers[headers.index('Pлин')] = 'Рлин'
    d = {}
    for i in range(len(headers)):
        if d.get(headers[i]) is not None:
            d[headers[i]] += 1
            headers[i] = headers[i] + '_' + str(d[headers[i]])
        else:
            d[headers[i]] = 1

    data.drop([0, 1], axis=0, inplace=True)

    data.drop([data.columns[0], data.columns[1]], axis=1, inplace=True)
    data.columns = headers
    data = data.reset_index(drop=True)
    for i in data.columns:
        data[i] = pd.to_numeric(data[i], errors='coerce')
    for i in range(len(data)):
        if (pd.isnull(data['КГФ'][i])) & (pd.notnull(data['КГФ_2'][i])):
            data['КГФ'][i] = data['КГФ_2'][i] * 1000
    data.drop('КГФ_2', axis=1, inplace=True)
    data.dropna(how="all", subset=['КГФ', 'G_total'], inplace=True)
    data = data.reset_index(drop=True)
    return data
