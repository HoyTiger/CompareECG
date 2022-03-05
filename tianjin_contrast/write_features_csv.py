from xml.dom import minidom
import os
import numpy as np
import torch
import csv
from net1d import Net1D
from numpy.linalg import norm
import pandas as pd

read_dir = '/data/0shared/shijia/对比范例'
save_dir = '/data/0shared/hanyuhu'
# pad_leads = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
# pad unit: 微伏 uV 
label_one = ['547653', '547959', '548115', '548119', '548821', '548902', '549462', '549648', '550132', '550166', '550172', '550415', '550873', '551099', '551123']

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
    df = pd.read_csv('/data/0shared/shijia/pair_distances_HR.csv')
    df.set_index('record', inplace=True)

    pair_names = get_first_level_subdirectories(read_dir)
    csv_file = open(f"{save_dir}/features_hr_label.csv", 'w+')
    csv_writer = csv.writer(csv_file)
    title = ['record_ind']
    for i in range(1, 1025):
        title.append(f'feature{i}')
    title.append('hr')
    title.append('label')
    csv_writer.writerow(title)
    for ind, pair_name in enumerate(pair_names):
        id = ind
        record = pair_name
        label = 0
        if  pair_name in label_one:
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

        hrs = df.loc[int(record)]

        record_ind0 = record + '_0'
        row1 = [record_ind0]
        for i in range(1024):
            row1.append(features_np[0][i])
        row1.append(hrs['HR_1'])
        row1.append(label)
        csv_writer.writerow(row1)

        record_ind1 = record + '_1'
        row2 = [record_ind1]
        for i in range(1024):
            row2.append(features_np[1][i])
        row2.append(hrs['HR_2'])
        row2.append(label)
        csv_writer.writerow(row2)
    csv_file.close()


