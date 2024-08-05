# Hybrid-Quantum-Classical-Computing
第十九届“挑战杯”全国大学生课外学术科技作品竞赛

## 作品名称
量子经典混合算法在抗菌肽设计与优化方面的应用及创新

## 队伍成员
王翊霖 姜肇瑞 陈亮锟 李闻天 肖雯心 谯望 闫泽玮 王笑  章一舟 王妍珊

## 指导老师
王志敏 崔鹏飞

## 摘要

做为一种新型的信息处理方法，量子计算基于量子力学揭示的不确定性和非局域性等基本原理进行信
息处理，在处理某些问题时，相比经典计算具有指数加速效果。量子深度学习作为量子计算与深度学习的
结合，处于前沿研究领域。它充分利用了量子计算机的高效计算能力和深度学习模型的复杂性，探索新的
计算范式，在处理高维数据、加速训练过程以及解决传统算法难题方面展示了突出的潜力。其面临大量数
据挑战时，量子计算与深度学习的结合有望降低深度学习成本，为深度学习算法的研究带来全新的灵感。

本模型基于机器学习方法实现对于抗菌肽的分析和预测。由于抗菌肽结构复杂，种类繁多，使得其序
列呈现出极度分散，长度差异性大，模型训练时，难以学习到其特征部分，效率低下，部分抗菌肽序列结
构无特殊性质但被错误学习等问题。针对以上问题，我们使用了 PC6 编码，它提供了肽序列中每个氨基酸
的六个理化性质，相较于 PC7、AC6、AC7 等编码方法表现出更高的精确度和特异性，且在外部测试数据
上的表现与内部测试数据相当，具有良好的泛化能力，高效的信息传递能力。同时我们剔除极小部分极度
不规则序列，并进行序列数据截断以最大限度保留其特征信息，达到更好的预测分析效果。

本文构造了量子长短期记忆神经网络(Quaantum Long Short-Term Memory，QLSTM）,用于抗菌肽理化
性质的预测。混合量子-经典长短期记忆神经网络主要由变分量子线路（Variational Quantum Circuit, VQC）
代替传统神经网络，门控机制通过量子门操作量子比特的叠加态和纠缠态，使其控制信息的流动方式更加
复杂且多样，且同时保留了长期依赖记忆的能力，旨在利用量子计算的特性来进一步优化记忆和信息处理
的效率。通过数值实验，在大样本中噪声的抗菌肽理化性质预测任务中，模型预测理化性质与其真实理化
性质数值高度吻合，在精确度，学习效率上均优于传统长短期记忆神经网络。

## 流程

![fig1](/fig/image.png)

## 数据
1.PC6编码的序列集  

2.序列长度小于四的理化性质集，部分数据做了归一化处理  
![fig2](/fig/fig.png)
## 模型
1.经典神经模型:  
![fig3](/fig/fig (2).png)  

编码后的数据首先进入LSTM层处理，取LSTM层的最后一位输出。随后再经过全连接层输出预测数据  

使用均方误差作为损失函数，Adam作为优化器，学习率设置为0.01

2.量子经典混合神经网络：
![fig4](/fig/fig(3).png)

编码后的数据进入QLSTM层处理，输出后的数据经过全连接层输出为预测数据

变分量子电路
![fig5](/fig/fig(4).png)

编码层部分使用双角编码法将数据编码为量子态。变分层使各个数据产生纠缠，随后Ansatz线路的旋转门旋转数据。测量层测量量子状态并将结果输出




