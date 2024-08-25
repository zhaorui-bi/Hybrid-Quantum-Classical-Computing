import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from deeplearning import Net
class  PC(Dataset):  #获得需要进行预测的序列编码后的张量
    def __init__(self,filepath=r"Sequence.json"):
        super().__init__()
        df = pd.read_json(filepath,dtype=np.float32)
        feat = df.iloc[0:, :]
        print(feat.shape)
        expanded_data = []
        for col in feat.columns:
            expanded_col = np.array(feat[col].tolist())
            expanded_data.append(expanded_col)
        data_values = np.array(expanded_data, dtype=np.float32)
        data_values = torch.from_numpy(data_values)
        self.data = data_values

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == '__main__':
    #加载训练完的11个网络并分别进行预测
    net1=torch.load("net1.plk")
    net2=torch.load("net2.plk")
    net3=torch.load("net3.plk")
    net4=torch.load("net4.plk")
    net5=torch.load("net5.plk")
    net6=torch.load("net6.plk")
    net7=torch.load("net7.plk")
    net8=torch.load("net8.plk")
    net9=torch.load("net9.plk")
    net10=torch.load("net10.plk")
    net11=torch.load("net11.plk")
    pc = PC()
    data = DataLoader(pc)
    prop = []
    hidden_prev = torch.zeros(2, 1, 64).requires_grad_().cuda()
    for seq in data:
        seq = seq.cuda()
        out1 = net1(seq,hidden_prev)
        out2 = net2(seq,hidden_prev)
        out3 = net3(seq,hidden_prev)
        out4 = net4(seq,hidden_prev)
        out5 = net5(seq,hidden_prev)
        out6 = net6(seq,hidden_prev)
        out7 = net7(seq,hidden_prev)
        out8 = net8(seq,hidden_prev)
        out9 = net9(seq,hidden_prev)
        out10 = net10(seq,hidden_prev)
        out11 = net11(seq,hidden_prev)
        prop.append(out1)
        prop.append(out2)
        prop.append(out3)
        prop.append(out4)
        prop.append(out5)
        prop.append(out6)
        prop.append(out7)
        prop.append(out8)
        prop.append(out9)
        prop.append(out10)
        prop.append(out11)
        #输出预测出的每一个序列对应的11个性质
        print(out1,out2,out3,out4,out5,out6,out7,out8,out9,out10,out11)

