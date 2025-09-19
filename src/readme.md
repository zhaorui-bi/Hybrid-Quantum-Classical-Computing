# 代码所需要的环境
Anaconda创建的虚拟环境，内部应当安装：

    - cuda版本的pytorch：版本为2.3.1+cu121
    - Pandas库
    - Numpy库
    - Matplotlib库
    - Pennylane库

# 代码使用方法
- ClassicLstm.py是经典网络，点击运行后会自动训练网络并保存。随后点开ClassicLstm_pred.py，用文件夹中的PC6编码代码将需要预测序列转换为PC6编码后将文件地址输入PC构造器中的filepath参数内，运行代码即可。代码会输出对应的十一个性质。

- QLSTM.py是混合神经网络，请在第332行处输入所需要预测的性质所在的列数（请注意，除了第7，8，11，12号性质，其他性质请用归一化后的数据运算）。

- 训练完成后代码会自动保存网络，随后调用即可。打开ClassicLstm_pred.py，将调用网络（第28到	第38行）代码中调用的plk文件换成保存的混合神经网络文件（经典的保存名为net，混合量子的保存名为model），其余操作同上。

# 数据逆归一化
    如果预测的是data中的physicoChemicalProperties[7]，physicoChemicalProperties[8]，physicoChemicalProperties[11],physicoChemicalProperties[12]之外的被归一化的性质请借用原数据库对改性质逆归一化以得到原始数据。如果是以上数据，则可以不用处理。