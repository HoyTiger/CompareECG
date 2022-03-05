import matplotlib.pyplot as plt

from QrsComplexDetector import *

import ast
import wfdb
from ecg_plot import plot_1

path = '/data/0shared/hanyuhu/ecg/data/ptbxl/'
sampling_rate = 500
# 读取文件并转换标签
Y = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))


def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path + f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path + f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data


# 获取原始信号数据
# X = load_raw_data(Y, sampling_rate, path)
for f in Y.filename_hr:
    X = np.array(wfdb.rdsamp(path +f)[0]).transpose(1,0)
    break
result = simple_adaptive_template_qrs_detect(X[0][:2500], sampling_rate)
plt.figure(figsize =(40,2))
plt.scatter(result.index, X[0][:2500][result.index], color='r')
plt.plot(X[0][:2500])
plt.show()