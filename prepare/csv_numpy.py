import os.path

import pandas as pd
import numpy as np
from xml.dom.minidom import parse


def calculate(df):
    df['III'] = df['II'] - df['I']
    df['aVR'] = - (df['II'] + df['I']) / 2
    df['aVL'] = (df['I'] - df['III']) / 2
    df['aVF'] = (df['II'] + df['III']) / 2
    return df


def transfer_csv2np(path, sava_path=None):
    print(path)
    df = pd.read_csv(path, header=1)
    if len(df.columns)!=8:
        return
    f, r = save_info(path)
    df = calculate(df)
    df = df[['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']]
    data = np.array(df) * r / 1000
    if sava_path is None:
        save_path = path.replace('.csv', '')
    else:
        if not os.path.exists(sava_path):
            os.makedirs(sava_path)
        basename = os.path.basename(path).replace('.csv', '')
        save_path = os.path.join(sava_path, basename)
    np.save(save_path, data)
    df.to_csv(path.replace('/Volumes/Seagate Basic/ecg_csv', '/Volumes/Seagate Basic/ecg_csv2'), index=False)


def save_info(csv_path):
    fn = open('/Volumes/Seagate Basic/ecg_csv2/info.txt', 'a+', encoding='utf-8')
    file = os.path.basename(csv_path)
    if csv_path.endswith('.csv'):
        with open(csv_path, 'r', encoding='utf-8') as f:
            line = f.readline()
            line = line.strip()
            l = line.split(',')
            fn.write(file + '\t' + l[1] + '\t' + l[4] + '\n')
            if float(l[1]) != 0.002 or float(l[4]) != 1.25:
                print(file + '\t' + l[1] + '\t' + l[4])
    fn.close()
    return float(l[1]), float(l[4])


def transfer_csv2np_xiamen(path, sava_path=None):
    print(path)
    df = pd.read_csv(path)

    f, r = save_info_xiamen(path)
    data = np.array(df) * r / 1000
    if sava_path is None:
        save_path = path.replace('.csv', '')
    else:
        if not os.path.exists(sava_path):
            os.makedirs(sava_path)
        basename = os.path.basename(path).replace('.csv', '')
        save_path = os.path.join(sava_path, basename)
    np.save(save_path, data)


def save_info_xiamen(csv_path):
    fn = open('/Volumes/Seagate Basic/厦门/info.txt', 'a+', encoding='utf-8')
    xml_path = csv_path.replace('.csv', '.xml')
    file = os.path.basename(csv_path)
    tree = parse(xml_path).documentElement
    r = tree.getElementsByTagName('resolution')[0].childNodes[0].data
    fre = tree.getElementsByTagName('samplingrate')[0].childNodes[0].data

    fn.write(file + '\t' + r + '\t' + fre + '\n')
    return float(fre),  float(r)


if __name__ == '__main__':
    path = '/Volumes/Seagate Basic/厦门/20220125012259'

    for file in os.listdir(path):
        if file.endswith('.csv'):
            filename = os.path.join(path, file)
            transfer_csv2np_xiamen(filename, os.path.join(os.path.dirname(path), 'NPY'))