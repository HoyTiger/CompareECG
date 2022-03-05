import torch
from net1d import Net1D
import numpy as np
from util.read_data import DataSet
import scipy.spatial.distance
import pandas as pd

from sklearn.metrics.pairwise import pairwise_distances

device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
model = Net1D(
    in_channels=12,
    base_filters=64,
    ratio=1,
    filter_list=[64, 160, 160, 400, 400, 1024, 1024],
    m_blocks_list=[2, 2, 2, 3, 3, 5, 5],
    kernel_size=16,
    stride=2,
    groups_width=16,
    verbose=False,
    use_bn=True,
    use_do=False,
    n_classes=47)

model.load_state_dict(torch.load('/data/0shared/shijia/diagnose_lead12.pth'), False)
# model.to(device)
model.eval()

dataset = DataSet('./20220207动静脉瘘患者XMLECG/').make_dataset()
d = []
for data in dataset:
    patient_name = data['patient_name']
    D = torch.Tensor(data['D'][np.newaxis,:, :5000])
    F = torch.Tensor(data['F'][np.newaxis,:, :5000])
    P = torch.Tensor(data['P'][np.newaxis,:, :5000])

    D_features = model(D)
    F_features = model(F)
    P_features = model(P)

    temp_map = {}
    temp_map['patient_name'] = patient_name

    for distance in ['cityblock', 'cosine', 'euclidean', 'l1', 'l2',
                     'manhattan']:
        d1 =pairwise_distances(D_features.detach().numpy(), F_features.detach().numpy(), metric=distance)
        d2 =pairwise_distances(P_features.detach().numpy(), F_features.detach().numpy(), metric=distance)
        d3 =pairwise_distances(P_features.detach().numpy(), D_features.detach().numpy(), metric=distance)

        temp_map[f'DF_{distance}'] = d1[0][0]
        temp_map[f'PF_{distance}'] = d2[0][0]
        temp_map[f'DP_{distance}'] = d3[0][0]

    d.append(temp_map)


df = pd.DataFrame(d)
df.to_csv('distance.csv')
