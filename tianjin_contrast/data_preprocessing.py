from xml.dom import minidom
import os
import numpy as np
import torch
import csv
from net1d import Net1D
from numpy.linalg import norm
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

read_dir = '/data/0shared/shijia/对比范例'
save_dir = '/data/0shared/hanyuhu/'
# pad_leads = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
# pad unit: 微伏 uV 
label_one = ['547653', '547959', '548115', '548119', '548821', '548902', '549462', '549648', '550132', '550166', '550172', '550415', '550873', '551099', '551123']
#hr_features = [530, 609, 370, 515, 465, 394, 934]
# hr_features = [530, 563, 480, 609, 482, 640, 497, 370, 515, 465, 394, 934, 
# 581, 643, 930, 216, 534, 934, 982, 431, 483, 533, 514, 302]


file = '/data/0shared/shijia/对比范例/features_hr_label.csv'
df = pd.read_csv(file)

X = df[df.columns[1:-2]]
y = df['hr']

# --- Extra-Trees
model_fit = ExtraTreesClassifier()
model_fit.fit(X, y)
# display the relative importance of each attribute
rank1 = np.argsort(model_fit.feature_importances_)
hr_features = rank1[12:]

model = Net1D(
        in_channels=12,
        base_filters=64,
        ratio=1,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        #m_blocks_list=[2, 2, 2, 3, 3, 5, 5],
        m_blocks_list=[2, 3, 3, 4, 4, 5, 5],
        kernel_size=16,
        stride=2,
        groups_width=16,
        verbose=False,
        use_bn=True,
        use_do=False,
        n_classes=47)

model.load_state_dict(torch.load('/data/0shared/shijia/diagnose_lead12.pth'))
#model.to(device)
model.eval()

def get_first_level_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if (name.startswith('.') is False and os.path.isdir(os.path.join(a_dir, name)))]

if __name__ == "__main__":
    pair_names = get_first_level_subdirectories(read_dir)
    csv_file = open(f"{save_dir}/pair_distances_no_hr_features_tree_12.csv", 'w+')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['id','record','distance','label'])
    for ind, pair_name in enumerate(pair_names):
        id = ind
        record = pair_name
        label = 0
        if pair_name in label_one:
            label = 1
        pair_dir = f"{read_dir}/{pair_name}"
        input_x_list = [] # a pair
        for filename in os.listdir(pair_dir): 
            data_12ch = []
            if filename.endswith('xml'):
                f = os.path.join(pair_dir, filename)
                mydoc = minidom.parse(f) # tianjin system1 pad
                for i in range(12):
                    ch = mydoc.getElementsByTagName(f'Ch{i}')
                    s = (ch[0].firstChild.data).split( )
                    sn = [int(ss) / 1000 for ss in s] # uV to mV
                    if len(sn) < 5000:
                        sn.extend([0] * (5000 - len(sn)))
                    data_12ch.append(sn[0:5000])         
                input_x_list.append(data_12ch)
        torch_list = torch.Tensor(input_x_list)
        features = model(torch_list)
        features_np = np.array(torch.Tensor.tolist(features))
        distance = norm(np.delete(features_np[0], hr_features) - np.delete(features_np[1],hr_features))
        csv_writer.writerow([id,record,distance,label])
    csv_file.close()