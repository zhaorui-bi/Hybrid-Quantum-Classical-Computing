import pandas as pd
import numpy as np
import torch
import torchvision
from torch.utils.data import random_split
from matplotlib import pyplot as plt
from collections import OrderedDict
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
# from torchsummary import summary

class CSVDataSet(Dataset):  #获取序列对应的某一列的性质
    def __init__(self,prop,filepath=r"data.csv"):
        print(f"读取 {filepath}")

        df = pd.read_csv(
            filepath, header=0, index_col=False,
            encoding='unicode_escape',
            usecols=[prop],
            dtype=np.float32,
            skip_blank_lines=True,
        )

        print(f"数据集的形状为 {df.shape}")
        feat = df.iloc[0:14067, 0:].values
        label = df.iloc[:, 0].values
        self.x = torch.from_numpy(feat)
        self.y = torch.from_numpy(label)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]


class  PC(Dataset):  #获取序列PC6编码后对应的张量
    def __init__(self,filepath=r"Sequence.json"):
        super().__init__()
        df = pd.read_json(filepath,dtype=np.float32)
        feat = df.iloc[:, 0:]
        print(feat.shape)
        expanded_data = []
        for col in feat.columns:
            expanded_col = np.array(feat[col].tolist())
            expanded_data.append(expanded_col)
        data_values = np.array(expanded_data, dtype=np.float32)
        data_values = torch.from_numpy(data_values)
        print(data_values.shape)
        self.data = data_values

    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


class Net(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(Net, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        for p in self.lstm.parameters():
            nn.init.normal_(p, 0.0, 0.001)
        self.linear = torch.nn.Linear(hidden_size, 1)

    def forward(self, x, hidden_prev):
        out, (hidden_prev,cell_prev) = self.lstm(x, (hidden_prev,hidden_prev))
        out = out[:, -1, :] # 取序列的最后一个输出作为预测结果
        # print(out)
        out = self.linear(out)
        return out, hidden_prev

class CombinedDataset(Dataset):  #将两个数据集合并成一个数据集
    def __init__(self, seq_dataset, feat_dataset):
        self.x = seq_dataset
        self.y = feat_dataset

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        seq = self.x[idx]
        feat = self.y[idx]
        return seq, feat

#训练lstm网络
def train(prop):
    input_data = PC()
    target_data = CSVDataSet(prop)
    assert len(input_data) == len(target_data)
    combined_dataset = CombinedDataset(input_data, target_data)
    # 划分训练集和测试集
    train_size = int(0.8 * len(combined_dataset))
    test_size = len(combined_dataset) - train_size
    train_dataset, test_dataset = random_split(combined_dataset, [train_size, test_size])
    data_loader_train = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    loss_func = torch.nn.MSELoss()
    loss_func = loss_func.cuda()
    net = Net(input_size=6, hidden_size=64, num_layers=2)
    net = net.cuda()
    optimizer =torch.optim.Adam(net.parameters(), lr=0.01)
    losses = []
    i=0
    output =[]
    target = []
    for epoch in range(10):
        for sequences, property in data_loader_train:
            optimizer.zero_grad()
            sequences = sequences.cuda()
            property=property.cuda()
            hidden_prev = torch.zeros(2, 64, 64).requires_grad_()  # 初始化隐藏状态
            hidden_prev = hidden_prev.cuda()
            output, hidden_prev = net(sequences,hidden_prev)
            loss = loss_func(output, property)
            hidden_prev.detach_()
            loss.backward()
            optimizer.step()
            #for name, parms in net.named_parameters():
            #   print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data),
            #         ' -->grad_value:', torch.mean(parms.grad))
            #print(loss.grad)
            # print(loss.item())
            if i %1 ==0 :
                losses.append(loss.item())
            i = i + 1
    plt.plot(losses, 'r')
    plt.xlabel('train')
    plt.ylabel('loss')
    plt.title('lstm')
    plt.show()
    plt.close()
    return net

#随机取百分之二十的数据测试
def test(net,prop):
    input_data = PC()
    target_data = CSVDataSet(prop)
    combined_dataset = CombinedDataset(input_data, target_data)
    # 划分训练集和测试集
    train_size = int(0.8 * len(combined_dataset))
    test_size = len(combined_dataset) - train_size
    train_dataset, test_dataset = random_split(combined_dataset, [train_size, test_size])
    data_loader_test= DataLoader(test_dataset, batch_size=1, shuffle=True, drop_last=True)
    loss_func = torch.nn.MSELoss()
    loss_func = loss_func.cuda()
    losses=[]
    outs = []
    propertys = []
    i=0
    with torch.no_grad():
        net.eval()
        for sequences, property in data_loader_test:
            sequences = sequences.cuda()
            property=property.cuda()
            hidden_prev = torch.zeros(2, 1, 64).requires_grad_()  # 初始化隐藏状态
            hidden_prev = hidden_prev.cuda()
            out,hidden_prev = net(sequences, hidden_prev)
            loss = loss_func(out, property)
            # print(out)
            # print(property)
            if i%100 ==0:
                losses.append(loss.item())
                out = out.item()
                property = property.item()
                propertys.append(property)
                outs.append(out)
            i = i+1
    #绘制拟合图
    plt.figure(figsize=(10, 5))
    plt.plot(outs,'b',label='Predicted')
    plt.plot(propertys, 'r',label='Actual')
    plt.xlabel('Sample index')
    plt.ylabel('Target value')
    plt.title('Test')
    plt.legend()
    plt.show()
    return losses

def model_structure(model):  #获取模型结构
    blank = ' '
    print('-' * 90)
    print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 90)
    num_para = 0
    type_size = 1  # 如果是浮点数就是4

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-' * 90)


if __name__ == '__main__':
    #训练出11个网络并保存
    net1 = train(3)
    print(test(net1,3))
    # model_structure(net1)
    net2 = train(5)
    test(net2,5)
    net3 = train(7)
    test(net3,7)
    net4 = train(9)
    test(net4,9)
    net5 = train(11)
    test(net5,11)
    net6 = train(13)
    test(net6,13)
    net7 = train(14)
    test(net7,14)
    net8 = train(15)
    test(net8,15)
    net9 = train(18)
    test(net9,18)
    net10 = train(19)
    test(net10,19)
    net11 = train(20)
    test(net11,20)
    torch.save(net1, "net1.plk")
    torch.save(net2, "net2.plk")
    torch.save(net3, "net3.plk")
    torch.save(net4, "net4.plk")
    torch.save(net5, "net5.plk")
    torch.save(net6, "net6.plk")
    torch.save(net7, "net7.plk")
    torch.save(net8, "net8.plk")
    torch.save(net9, "net9.plk")
    torch.save(net10, "net10.plk")
    torch.save(net11, "net11.plk")



