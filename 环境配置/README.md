##  wsl与torch（pytorch，torch quantum）的安装教程

#### 1.wsl的安装

1.win+r打开输入面板，输入cmd   

2.输入wsl -install   

3.等待安装，完成后根据提示重启电脑   

4.重启后会要求创建账户和密码，记住自己自己的密码（之后要用），   

密码输入的时候**不会显示**，但是不用担心，你已经输进去了   

#### 2.Anaconda和cuda

1.windows版的搜索anaconda和cuda即可，安装anaconda时记住一定要**勾选那个添加路径**   

<em>那个选项有个PATH的单词而且是不推荐的，别管，点</em>    

   [CUDA下载地址]([CUDA Toolkit 12.5 Downloads | NVIDIA Developer](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local) "CUDA下载")   

下载的东西如下图   

![图片1](/环境配置/1.png)

**记住下载的位置**

#### 3.添加路径

先打开cmd，输入conda和nvcc –V。如果显示不是内部指令做下列步骤，如果显示了一大堆

跳过本步

1.桌面右键，点击个性化，在搜索框输入高级系统设置                               

点击环境变量进入，打开Path

![图片2](/环境配置/2.png)

![图片4](/环境配置/3.png)

**Anaconda**需要的路径如下：

![图片5](/环境配置/5.png)

**Cuda**需要的路径如下：

![图片3](/环境配置/3.png)

<em>配置完成如下图</em>

![图片6](/环境配置/6.png)

![图片7](/环境配置/7.png)

#### 4.Ubuntu anaconda cuda安装

<em>小tips：一定记得cd到根目录</em>>

1.anaconda按照流程里的此网站来就可：

[anaconda配置教程](https://blog.csdn.net/weixin_44878336/article/details/133967607)

额外tips：配置文件那里打开配置后往下翻到最下面直到你看到initial，把下面的命令输进去回车，关掉重开。接着做就可以了

2.**cuda**<em>(难点)</em>:

[CUDA下载地址]([CUDA Toolkit 12.5 Downloads | NVIDIA Developer](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local) "CUDA下载")  

![图片8](/环境配置/8.png)

 按照上图找到安装命令<em> 白色的框表示选中</em>

1.（tips：记住是在**wsl**里）依次输入命令框里的命令下载

2.配置环境：

输入sudo nano /home/$USER/.bashrc

和anaconda配置一样拉到最底部（<em>你应该能看到一行initial</em>），输入

export CUDA_HOME=/usr/local/cuda

export PATH=$PATH:$CUDA_HOME/bin

export LD_LIBRARY_PATH=/usr/local/cuda-12.5/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

**第三行的cuda-12.5这里依据下载的版本填写**

更新文件

source /home/$USER/.bashrc

更新一下可能需要的依赖（也可能不需要，先试试最后一行指令，不行再运行这行）：

sudo apt-get install freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libgl1-mesa-glx libglu1-mesa libglu1-mesa-dev

nvcc –V,没报错就大功告成

配置操作如下打开后复制路径如下：<em>然后按住ctrl+x，按yes，然后回车即可。</em>

#### 5.pytorch

<<em>依然是在wsl'里</em>>

conda create –n <em>虚拟环境的名字</em>  python=3.8  创建环境

activate conda <em>虚拟环境的名字</em>

![图片9](/环境配置/9.png)

官网将这段指令粘到启动的虚拟环境里                                                                                                                           cuda版本要匹配:cmd里面输入nvidai-smi可查看版本

![图片10](/环境配置/10.png)

安装完成后输入python

再输入import torch

输入torch.cuda.is_available

如果显示True则安装完成

#### 6.Torch quantum

1.cd到根目录

2.git clone https://github.com/mit-han-lab/torchquantum.git

cd torchquantum

pip install --editable .依次输入

**如果运行最后一步时出现了红色的error报错，请再运行一次最后一句**

#### 测试<依旧是wsl里>

前往torch quantum官方下载好example

1.cd到example是所在位置

2.输入python example.py

若可以成功运行用pycharm打开example，修改下面这一行**cpu**改成**cuda**

![图片11](/环境配置/11.png)

再运行一次

![图片12](/环境配置/12.png)

如果结果这里显示**device：cuda**

## 恭喜*★,°*:.☆\(￣▽￣)/$:*.°★* 

你已经成功配置完所需的环境了。



 

