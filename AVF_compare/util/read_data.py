import os
from xml.dom import minidom

import numpy as np



def read_data(path):
    data = []
    mydoc = minidom.parse(path)  # tianjin system1 pad
    for i in range(12):
        ch0 = mydoc.getElementsByTagName(f'Ch{i}')
        s = (ch0[0].firstChild.data).split()
        sn = [int(ss) / 1000 for ss in s]
        data.append(sn)

    return np.array(data)

class DataSet():
    def __init__(self, dirname):
        self.dirname = dirname
        self.file_list = os.listdir(dirname)

    def len(self):
        return len(self.file_list)

    def make_dataset(self):
        dataset = []
        for index in range(self.len()):
            patient_name = self.file_list[index]

            path = os.path.join(self.dirname, patient_name)
            if os.path.isfile(path):
                continue
            files = os.listdir(path)
            data = {}
            for file in files:
                xml_path = os.path.join(path, file)
                if file.endswith('_D.xml'):
                    data['D'] = read_data(xml_path)
                elif file.endswith('_F.xml'):
                    data['F'] = read_data(xml_path)
                else:
                    data['P'] = read_data(xml_path)
            data['patient_name'] = patient_name
            dataset.append(data)
        return dataset







if __name__ == '__main__':
    # dataset('../20220207动静脉瘘患者XMLECG/')
    data = DataSet('../20220207动静脉瘘患者XMLECG/')
    dataset = data.make_dataset()

