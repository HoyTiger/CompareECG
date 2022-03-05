import torch 
from net1d import Net1D
import numpy as np

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
#model.to(device)
model.eval()
input_x = np.load('/data/0shared/shijia/MEDEXSKFK2019061716153174a.npy')[:,0:5000]
input_x_list = []
input_x_list.append(input_x)
print(input_x.shape)
torch_list = torch.Tensor(input_x_list)
features = model(torch_list)
print(torch_list.shape)
print(features.shape)  