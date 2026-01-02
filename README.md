# 🧬 量子经典混合算法在抗菌肽设计与优化方面的应用及创新

### Application and Innovation of Hybrid Quantum-Classical Algorithms in Antimicrobial Peptide Design and Optimization

> **第十九届“挑战杯”全国大学生课外学术科技作品竞赛参赛作品**

---

## 👥 团队介绍 (Team)

| 角色 | 成员名单 |
| --- | --- |
| **队伍成员** | 王翊霖, 姜肇瑞, 陈亮锟, 谯望, 闫泽玮, 王妍珊, 王笑, 肖雯心, 章一舟, 李闻天 |
| **指导老师** | 王志敏, 崔鹏飞 |

---

## 📖 摘要 (Abstract)

作为一种新型的信息处理方法，**量子计算**基于量子力学揭示的不确定性和非局域性等基本原理进行信息处理，在处理某些问题时，相比经典计算具有指数加速效果。**量子深度学习**作为量子计算与深度学习的结合，处于前沿研究领域。它充分利用了量子计算机的高效计算能力和深度学习模型的复杂性，探索新的计算范式，在处理高维数据、加速训练过程以及解决传统算法难题方面展示了突出的潜力。其面临大量数据挑战时，量子计算与深度学习的结合有望降低深度学习成本，为深度学习算法的研究带来全新的灵感。

本项目基于机器学习方法实现对于**抗菌肽（Antimicrobial Peptides, AMPs**的分析和预测。

* **挑战**：由于抗菌肽结构复杂，种类繁多，序列呈现出极度分散且长度差异大，导致传统模型难以提取特征，效率低下。
* **创新**：
1. **PC6 编码**：使用了 PC6 编码，提供了肽序列中每个氨基酸的六个理化性质。相较于 PC7、AC6、AC7 等编码方法，表现出更高的精确度、特异性和泛化能力。
2. **数据处理**：剔除极小部分极度不规则序列，并进行序列数据截断以最大限度保留其特征信息。
3. **QLSTM 模型**：构造了 **量子长短期记忆神经网络 (Quantum LSTM)**。由变分量子线路（VQC）代替传统神经网络层。门控机制通过量子门操作量子比特的叠加态和纠缠态，使控制信息的流动方式更加复杂且多样。



通过数值实验，在大样本中噪声的抗菌肽理化性质预测任务中，本模型在精确度和学习效率上均**优于传统长短期记忆神经网络**。

---

## 🔄 流程 (Workflow)

![fig1](/fig/image.png)
---

## 📊 数据处理 (Data)

本项目采用的数据集包含以下特征：

1. **PC6 编码序列集**：利用氨基酸的理化性质进行高维编码。
2. **预处理**：
* 剔除序列长度小于 4 的数据。
* 对部分理化性质数据进行了**归一化 (Normalization)** 处理。

![fig2](/fig/fig.png)


---

## 🧠 模型架构 (Architecture)

![fig3](/fig/fig2.png)  

### 1. 经典神经网络模型 (Classical LSTM)

* **结构**：编码后的数据  LSTM 层  取最后一位输出  全连接层 (FC)  预测结果。
* **设置**：
* 损失函数：均方误差 (MSE)
* 优化器：Adam
* 学习率：



### 2. 量子经典混合神经网络 (Hybrid QLSTM)

![fig4](/fig/fig3.png)

* **结构**：编码后的数据  **QLSTM 层**  全连接层 (FC)  预测结果。

#### ⚛️ 变分量子电路 (Variational Quantum Circuit, VQC)

![fig5](/fig/fig4.png)

VQC 是 QLSTM 的核心组件，其处理流程如下：

1. **编码层 (Encoding Layer)**：使用双角编码法 (Angle Embedding) 将经典数据映射为量子态。
2. **变分层 (Variational Layer)**：
* 使用 CNOT 门使各个量子比特产生纠缠。
* Ansatz 线路通过旋转门调整参数。


3. **测量层 (Measurement Layer)**：测量量子状态并将期望值输出回经典神经网络。

---

## 📜 许可证 (License)

本项目代码和模型基于 **GNU General Public License v3.0** 发布。
您可以免费用于学术研究和商业目的，但需遵守 GPL v3.0 协议。

Our model and code are released under GNU General Public License v3.0, and can be freely used for both academic and commercial purposes.

---

## 🖊️ 引用 (Citation)

如果您在研究中使用了本项目的代码或模型，请引用以下内容：

```bibtex
@misc{wang2024hybrid,
  title  = {Application and Innovation of Hybrid Quantum-Classical Algorithms in Antimicrobial Peptide Design and Optimization},
  author = {Wang, Yilin and Jiang, Zhaorui and Chen, Liangkun and Qiao, Wang and Yan, Zewei and Wang, Yanshan and Wang, Xiao and Xiao, Wenxin and Zhang, Yizhou and Li, Wentian},
  year   = {2024},
  note   = {The 19th "Challenge Cup" National College Student Extracurricular Academic Science and Technology Works Competition},
  url    = {https://github.com/yourusername/your-repo-name}
}

```
