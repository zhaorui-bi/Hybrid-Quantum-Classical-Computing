from torch.nn.parameter import Parameter
import numpy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from pennylane import ThermalRelaxationError
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
from torch import Tensor
import pennylane as qml
from pennylane import transforms
# 设置条件
c0 = qml.noise.op_eq(qml.PauliX) | qml.noise.op_eq(qml.PauliY)
c1 = qml.noise.op_eq(qml.Hadamard) & qml.noise.wires_in([0, 1])
c2 = qml.noise.op_eq(qml.RX)

@qml.BooleanFn
def c3(op, **metadata):
    return isinstance(op, qml.RY) and op.parameters[0] >= 0.5

# 设置含噪声的操作
n0 = qml.noise.partial_wires(qml.AmplitudeDamping, 0.4)

def n1(op, **metadata):
    ThermalRelaxationError(0.4, metadata["t1"], 0.2, 0.6, op.wires)

def n2(op, **metadata):
    qml.RX(op.parameters[0] * 0.05, op.wires)

n3 = qml.noise.partial_wires(qml.PhaseDamping, 0.9)

# 建立噪声模型
noise_model = qml.NoiseModel({c0: n0, c1: n1, c2: n2}, t1=0.04)
noise_model += {c3: n3}  # 逐一构建

class QLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, device, n_qubits=4, n_qlayers=1, batch_first=True, return_sequences=False, return_state=False, backend="default.mixed"):
        super(QLSTM, self).__init__()
        factory_kwargs = {'device': device, 'dtype': numpy.dtype}
        self.n_inputs = input_size
        self.hidden_size = hidden_size
        self.concat_size = self.n_inputs + self.hidden_size
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.backend = backend
        self.param1 = Parameter(torch.rand(3, 4))
        self.param2 = Parameter(torch.rand(3, 4))
        self.param3 = Parameter(torch.rand(3, 4))
        self.param4 = Parameter(torch.rand(3, 4))  #初始化量子参数
        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.device = device

        self.wires_forget = [f"wire_forget_{i}" for i in range(self.n_qubits)]
        self.wires_input = [f"wire_input_{i}" for i in range(self.n_qubits)]
        self.wires_update = [f"wire_update_{i}" for i in range(self.n_qubits)]
        self.wires_output = [f"wire_output_{i}" for i in range(self.n_qubits)]
        self.wires_add = [f"wire_add_{i}" for i in range(self.n_qubits)]

        self.dev_forget = qml.device(self.backend, wires=self.wires_forget)
        self.dev_input = qml.device(self.backend, wires=self.wires_input)
        self.dev_update = qml.device(self.backend, wires=self.wires_update)
        self.dev_output = qml.device(self.backend, wires=self.wires_output)
        self.dev_add = qml.device(self.backend, wires=self.wires_add)

        self.linear = nn.Linear(self.hidden_size, 1)

        def ansatz(params, wires_type):
            # 纠缠层
            for i in range(1, 3):
                for j in range(self.n_qubits):
                    if j + i < self.n_qubits:
                        qml.CNOT(wires=[wires_type[j], wires_type[j + i]])
                    else:
                        qml.CNOT(wires=[wires_type[j], wires_type[j + i - self.n_qubits]])

            # 变分层
            for i in range(self.n_qubits):
                qml.RX(params[0][i], wires=wires_type[i])
                qml.RY(params[1][i], wires=wires_type[i])
                qml.RZ(params[2][i], wires=wires_type[i])

        def add_noise_to_qnode(qnode, noise_model):
            return transforms.add_noise(qnode, noise_model)

        # 定义电路
        def _circuit_forget(inputs, weights):
            qml.Hadamard(wires=self.wires_forget.__getitem__(0))
            qml.Hadamard(wires=self.wires_forget.__getitem__(1))
            qml.Hadamard(wires=self.wires_forget.__getitem__(2))
            qml.Hadamard(wires=self.wires_forget.__getitem__(3))
            qml.templates.AngleEmbedding(Tensor.arctan(inputs), wires=self.wires_forget,rotation="Y")
            qml.templates.AngleEmbedding(Tensor.arctan(inputs*inputs), wires=self.wires_forget,rotation="Z")
            qml.templates.BasicEntanglerLayers(weights, wires=self.wires_forget)
            ansatz(self.param1, self.wires_forget)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_forget]

        self.qlayer_forget = qml.QNode(_circuit_forget, self.dev_forget, interface="torch")
        self.qlayer_forget = add_noise_to_qnode(self.qlayer_forget, noise_model)

        def _circuit_input(inputs, weights):
            qml.Hadamard(wires=self.wires_input.__getitem__(0))
            qml.Hadamard(wires=self.wires_input.__getitem__(1))
            qml.Hadamard(wires=self.wires_input.__getitem__(2))
            qml.Hadamard(wires=self.wires_input.__getitem__(3))
            qml.templates.AngleEmbedding(Tensor.arctan(inputs),wires=self.wires_input,rotation="Y")
            qml.templates.AngleEmbedding(Tensor.arctan(inputs*inputs),wires=self.wires_input,rotation="Z")
            qml.templates.BasicEntanglerLayers(weights, wires=self.wires_input)
            ansatz(self.param2, self.wires_input)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_input]

        self.qlayer_input = qml.QNode(_circuit_input, self.dev_input, interface="torch")
        self.qlayer_input = add_noise_to_qnode(self.qlayer_input, noise_model)

        def _circuit_update(inputs, weights):
            qml.Hadamard(wires=self.wires_update.__getitem__(0))
            qml.Hadamard(wires=self.wires_update.__getitem__(1))
            qml.Hadamard(wires=self.wires_update.__getitem__(2))
            qml.Hadamard(wires=self.wires_update.__getitem__(3))
            qml.templates.AngleEmbedding(Tensor.arctan(inputs), wires=self.wires_update,rotation="Y")
            qml.templates.AngleEmbedding(Tensor.arctan(inputs*inputs),wires=self.wires_update,rotation="Z")
            qml.templates.BasicEntanglerLayers(weights, wires=self.wires_update)
            ansatz(self.param3, self.wires_update)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_update]

        self.qlayer_update = qml.QNode(_circuit_update, self.dev_update, interface="torch")
        self.qlayer_update = add_noise_to_qnode(self.qlayer_update, noise_model)

        def _circuit_output(inputs, weights):
            qml.Hadamard(wires=self.wires_output.__getitem__(0))
            qml.Hadamard(wires=self.wires_output.__getitem__(1))
            qml.Hadamard(wires=self.wires_output.__getitem__(2))
            qml.Hadamard(wires=self.wires_output.__getitem__(3))
            qml.templates.AngleEmbedding(Tensor.arctan(inputs), wires=self.wires_output,rotation="Y")
            qml.templates.AngleEmbedding(Tensor.arctan(inputs*inputs), wires=self.wires_output,rotation="Z")
            qml.templates.BasicEntanglerLayers(weights, wires=self.wires_output)
            ansatz(self.param4, self.wires_output)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_output]

        self.qlayer_output = qml.QNode(_circuit_output, self.dev_output, interface="torch")
        self.qlayer_output = add_noise_to_qnode(self.qlayer_output, noise_model)

        def _circuit_add(inputs,weights):
            qml.Hadamard(wires=self.wires_update.__getitem__(0))
            qml.Hadamard(wires=self.wires_update.__getitem__(1))
            qml.Hadamard(wires=self.wires_update.__getitem__(2))
            qml.Hadamard(wires=self.wires_update.__getitem__(3))
            qml.templates.AngleEmbedding(Tensor.arctan(inputs), wires=self.wires_add,rotation="Y")
            qml.templates.AngleEmbedding(Tensor.arctan(inputs*inputs),wires=self.wires_add,rotation="Z")
            qml.templates.BasicEntanglerLayers(weights, wires=self.wires_add)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_add ]

        self.qlayer_add = qml.QNode(_circuit_add, self.dev_add, interface="torch")
        self.qlayer_add = add_noise_to_qnode(self.qlayer_add, noise_model)

        weight_shapes = {"weights": (n_qlayers, n_qubits)}
        print(f"weight_shapes = (n_qlayers, n_qubits) = ({n_qlayers}, {n_qubits})")

        self.clayer_in = torch.nn.Linear(self.concat_size, n_qubits)
        self.VQC = {
            'forget': qml.qnn.TorchLayer(self.qlayer_forget, weight_shapes),
            'input': qml.qnn.TorchLayer(self.qlayer_input, weight_shapes),
            'update': qml.qnn.TorchLayer(self.qlayer_update, weight_shapes),
            'output': qml.qnn.TorchLayer(self.qlayer_output, weight_shapes),
            'add': qml.qnn.TorchLayer(self.qlayer_add, weight_shapes)
        }  #定义VQC模块
        self.clayer_out = torch.nn.Linear(self.n_qubits, self.hidden_size)

    def forward(self, x, init_states=None):
        if self.batch_first is True:
            batch_size, seq_length, features_size = x.size()
        else:
            seq_length, batch_size, features_size = x.size()

        hidden_seq = []
        if init_states is None:
            h_t = torch.zeros(batch_size, self.hidden_size).to(self.device)  # 隐藏状态
            c_t = torch.zeros(batch_size, self.hidden_size).to(self.device)  # 细胞状态
        else:
            h_t, c_t = init_states.to(self.device)
            h_t = h_t[0]
            c_t = c_t[0]

        for t in range(seq_length):
            x_t = x[:, t, :]
            # 合并输入和隐藏状态
            v_t = torch.cat((h_t, x_t), dim=1)
            # 匹配量子维度
            y_t = self.clayer_in(v_t)

            f_t = torch.sigmoid(self.clayer_out(self.VQC['forget'](y_t)))  # 遗忘模块
            i_t = torch.sigmoid(self.clayer_out(self.VQC['input'](y_t)))  # 输入模块
            g_t = torch.tanh(self.clayer_out(self.VQC['update'](y_t)))  # 更新模块
            o_t = torch.sigmoid(self.clayer_out(self.VQC['output'](y_t)))  # 输出模块

            c_t = (f_t * c_t) + (i_t * g_t)
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        output = self.linear(h_t)  #通过最后一个时间步的输出进行预测
        return output

class CSVDataSet(Dataset): #获取序列对应的某一列的性质
    def __init__(self,prop,num,filepath=r"data.csv"):
        #print(f"读取 {filepath}")

        df = pd.read_csv(
            filepath, header=0, index_col=False,
            encoding='unicode_escape',
            usecols=[prop],
            dtype=np.float32,
            skip_blank_lines=True,
        )

        #print(f"数据集的形状为 {df.shape}")
        feat = df.iloc[0:num, 0:].values
        label = df.iloc[:, 0].values
        self.x = torch.from_numpy(feat)
        self.y = torch.from_numpy(label)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]


class  PC(Dataset):  #获取序列PC6编码后对应的张量
    def __init__(self,num,filepath=r"Sequence.json"):
        super().__init__()
        df = pd.read_json(filepath,encoding='unicode_escape',dtype=np.float32)
        feat = df.iloc[0:, 0:num]
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

class CombinedDataset(Dataset):
    def __init__(self, seq_dataset, feat_dataset):
        self.x = seq_dataset
        self.y = feat_dataset

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        seq = self.x[idx]
        feat = self.y[idx]
        return seq, feat

#逆归一化还原数据
def InverseNormalization(x,prop):
    Max=max(CSVDataSet(prop,14067)).item()
    Min=min(CSVDataSet(prop,14067)).item()
    output=(Max-Min)*x+Min
    return output

#训练Qlstm网络
def trainer(model, epoch, learning_rate, device):
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    model = model.to(device)
    train_loss_list = []
    for e in range(epoch):
        all_y_pred = []
        all_y = []
        i=0
        for x, y in dataloader:
            i+=1
            x = x.to(device)
            no_zero = x !=0
            x = x[no_zero]
            x = x.view(1,-1,6)
            y = y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i%10==0:   #收集数据绘图
                y=InverseNormalization(y, prop)
                print(y)
                y_pred=InverseNormalization(y_pred,prop)
                print(y_pred)
                y_pred_np = y_pred.cpu().detach().numpy()
                y_np = y.cpu().detach().numpy()
                all_y_pred.append(y_pred_np)
                all_y.append(y_np)

        # 合并数据
        all_y_pred = np.concatenate(all_y_pred, axis=0)
        all_y = np.concatenate(all_y, axis=0)
        loss_np = loss.cpu().detach().numpy()
        train_loss_list.append(loss_np)
        print("Epoch: {} loss {}".format(e, loss_np))

        # 绘制拟合图
        plt.figure(figsize=(10, 5))
        plt.plot(all_y, label='Actual')
        plt.plot(all_y_pred, label='Predicted')
        plt.title(f'Epoch {e} ')
        plt.ylabel('Target value')
        plt.legend()
        plt.show()
        #plt.savefig(f'epoch_{e}_plot.png')  # 保存图像
        plt.close()

    metrics = {'epoch': list(range(1, len(train_loss_list) + 1)),
               'train_loss': train_loss_list}
    return model, metrics


if __name__ == '__main__':
    input_size = 6
    hidden_size = 3
    output_size = 1
    # 创建模型实例
    prop = 5
    device = torch.device("cuda")
    model = QLSTM(input_size, hidden_size, device)
    input_data = PC(1000)
    target_data = CSVDataSet(prop,1000)
    assert len(input_data) == len(target_data)
    combined_dataset = CombinedDataset(input_data, target_data)
    dataloader = DataLoader(combined_dataset, batch_size=1, shuffle=False)
    lr = 0.01  # 定义学习率为0.01
    epoch = 5  # 定义迭代次数
    device = torch.device("cuda")
    optim_model, metrics = trainer(model, epoch, lr, device)
    #torch.save(optim_model, "net.pkl")
    epoch = metrics['epoch']
    loss = metrics['train_loss']

    # 创建图和Axes对象
    # 绘制训练损失曲线
    plt.plot(epoch, loss, label='Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    plt.legend()



