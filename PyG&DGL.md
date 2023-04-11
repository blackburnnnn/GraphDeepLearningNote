### 环境配置

- (12/5)查看Ubuntu版本 `cat /proc/version`

- 当前所有图节点异常检测代码的路径

  <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20221205235005629.png" alt="image-20221205235005629" style="zoom:77%;" />

- 之后本机就在这个文件夹下git clone，然后拖到远程服务器对应位置，因为直接clone的话，Pycharm也传不过去

- CUDA是显卡厂商NVIDIA推出的只能用于自家GPU的并行计算框架，只有安装这个框架才能够进行复杂的并行计算；目前主流的深度学习框架也都是基于CUDA进行GPU并行加速的；cuDNN是针对深度卷积神经网络的加速库；用python代码都可以验证两者是否可用

- (12/12)[DGL安装](https://blog.csdn.net/weixin_43918046/article/details/122181896?spm=1001.2014.3001.5501)

  - 建议安装最新版本，但是我跑代码需要不同的dgl版本，可以在同一个虚拟环境中卸载再安装不同版本的DGL

  - 当前Ubuntu版本 <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20221212184959771.png" alt="image-20221212184959771" style="zoom:33%;" />

  - 在自己的Ubuntu(无GPU)上安装DGL就不需要指定CUDA信息 `conda install -c dglteam dgl`

    <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20221212184801183.png" alt="image-20221212184801183" style="zoom:30%;" />

  - 以上默认命令将安装 `dgl-0.9.1post1-py37_0` 和 `networkx-2.2-py37_1`

    <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20221212185059230.png" alt="image-20221212185059230" style="zoom:43%;" />

  - **networkx是用来图可视化的，PyG和DGL都会使用到**

  - pytorch版本 <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20221213191627306.png" alt="image-20221213191627306" style="zoom:80%;" />

  - 更新scipy `pip install scipy --upgrade`

  - [pip list是conda list的子集](https://blog.csdn.net/nyist_yangguang/article/details/111304014)，**所以查看当前虚拟环境包的版本号，应输入`pip list`**

  - dgl和networkx配套下载成功 <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20221213192703740.png" alt="image-20221213192703740" style="zoom:70%;" />

- [图神经网络基础DGL版本](https://www.bilibili.com/video/BV1U44y1K7yP?p=1&vd_source=726461adc26f0b0f56256c07f5a478dc)

- (12/14)其实不需要一个项目一个项目手动进行远程路径映射，只要都在一个父项目里面，把子项目拖到远程服务器就可以自动进行映射了

  <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20221214104732085.png" alt="image-20221214104732085" style="zoom:67%;" />

- 目前我的pytorch和cuda不兼容 <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20221214152456167.png" alt="image-20221214152456167" style="zoom:50%;" />，之后需要安装cuda版本的pytorch

  ```python
  print(torch.__version__)
  print(torch.cuda.is_available())
  ```



### PyG

#### 唐宇迪教程

> [b站链接](https://www.bilibili.com/video/BV1FK411m7Ac/?spm_id_from=333.1007.top_right_bar_window_custom_collection.content.click&vd_source=726461adc26f0b0f56256c07f5a478dc)

##### 数据集&邻接矩阵

- (11/9)对于空手道数据集任务而言，空手道俱乐部34个会员作为节点，每个节点特征向量为34维，社交关系共78条边；具体任务是**对34个节点进行分类**

- VSCode更换jupyter内核

  <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20221109105509835.png" alt="image-20221109105509835" style="zoom:60%;" />

- 数据导入后只有一张图，用`dataset[0]`表示，主要的属性有：节点种类数 `num_classes`、节点特征数 `num_features`

- 再强调一次，PyG节点之间的连接关系必然用2×\{连接关系数}的二维矩阵 `edge_index` 确定，这个二维矩阵的第一行就是被连接点，第二行就是连接，`edge_index` 的类型是Tensor，**张量要进行转置直接 `.t()`**

  ```python
  source [[3,3,2,9],
  target [1,4,2,0]]
  ```

- `print dataset[0]`  <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20221109111603734.png" alt="image-20221109111603734" style="zoom:80%;" />

  - y的维度是34，就是类别数；train_mask就是会自动选择有标签的样本作为训练样本，这里34个节点都有标签，全都参与训练

- 反正都是转化成PyG的Data格式放入模型，具体的Data细节可以参看[官方文档](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html)

##### 模型定义&训练流程

- `from torch_geometric.utils import to_networkx` 可视化工具，之后需要学会将原始的实际数据转化为适配PyG的Data格式，图用轮子画就完事儿了
- **大佬都告诉我GCN等等创始性的理论文章就没必要深究了，直接看代码，那就冲！**

- class **GCN**(torch.nn.Module)

  1. 老样子继承 `torch.nn.Module`，套路写一行 `super().__init__()`，设置一下随机种子

  2. (11/10)然后就是堆层，后一层的输入维度需要等于前一层的输出维度，结合博客教程和官方文档的GCNConv层用法后知道，GCNConv的输入参数需要有输入数据和邻接矩阵两个，邻接矩阵格式不再说，**输入数据的维度都是(n,f_num)，n个数据，每个数据f_num个特征**，**GCNConv(其他类型的层肯定也都是这样！)在模型定义时，输入输出的维度，仅考虑节点的特征数！有多少个节点，会自动进行处理**；像如下代码里面，不管来了多少个节点，记作k，每个节点的特征数是dataset.num_feature，那么经过这四个层，最终就得到了**k个特征数为dataset.num_classes**，其实线性变换也是一种映射，把最后每个节点都映射到维度为num_classes的一维向量上，哪个位置的值越大，分类就过就是那个类

     ```python
     self.conv1 = GCNConv(dataset.num_features, 4)
     self.conv2 = GCNConv(4, 4)
     self.conv3 = GCNConv(4, 2)
     self.classifier = Linear(2, dataset.num_classes)
     ```

  3. 前向传播函数如下，**主函数中实例化好model后，直接通过 `model(x)` 获取输出，这个x包含.data属性和.edge_index属性，代码会自动处理的！**

     ```python
     def forward(self, x, edge_index):
       h = self.conv1(x, edge_index)  # 输入特征与邻接矩阵（注意格式，上面那种）
       h = h.tanh()
       h = self.conv2(h, edge_index)
       h = h.tanh()
       h = self.conv3(h, edge_index)
       h = h.tanh()
       out = self.classifier(h)  # 分类层
       return out, h
     ```

  4. 训练步骤再总结

     1. 实例化模型类

     2. 定义**优化器&损失函数**

     3. `for epoch in n` 中进行 `train(data) `，**每个epoch中的`train`方法如下**，复习这一二三步的顺序，不能颠倒；注意PyG Data中 `.train_mask` 就是用来标记哪些数据有标签

        ```python
        optimizer.zero_grad() # 第1步：梯度置0
        out, h = model(data.x, data.edge_index) # 直接model(x)应该也可以
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward() # 第2步：计算得到loss后，再loss.backward()，保存梯度
        optimizer.step() # 第3步：根据梯度更新参数
        ```

- 之后建模的时候，输入都是节点和邻接矩阵(`edge_index`)

##### GNN深入理解

- [图技术在美团外卖下的场景化应用及探索](https://tech.meituan.com/2022/09/08/gnn-scenariomodeling-subgraphextend-jointtraining.html)

- 应用场景：推荐系统；欺诈检测；知识图谱；道路交通流量预测

- 最后做的事无外乎对节点或边或整个图做分类、回归

- 现在说的GNN和传统深度学习，是因为输入数据格式不固定，但是现在跑的代码，输入的数据都是固定结构，所以说还不明确

- GCN和CNN是完全不同的，只是叫法相似

- (11/11)用softmax激活函数，只是人们直观上提供了一个概率的东西，但是在反向传播的过程中，**终究是根据每一次迭代的结果和 实际标签的差值进行参数的更新**

- [图神经网络层的本质](https://www.bilibili.com/video/BV1FK411m7Ac?p=11&vd_source=726461adc26f0b0f56256c07f5a478dc)

  1. **无论是GCN还是其他图神经网络层，不管搭多少层，都只是去聚合、改变图中节点的特征，另外邻接矩阵永远不变**
  2. **图神经网络层一般不搭深层，因为都是聚合邻居信息，搭深了没意义**
  3. **归根结底是利用图结构得到特征，最终基于特征去落地具体的场景**

- 邻接矩阵就是为了和特征矩阵打配合，**同时如果没有度矩阵，就不能引入节点自身的信息！因为对角线元素为0**

  <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20221111112330744.png" alt="image-20221111112330744" style="zoom:40%;" />

- [这个视频](https://www.bilibili.com/video/BV1FK411m7Ac?p=12&vd_source=726461adc26f0b0f56256c07f5a478dc)就解释了**为什么要左右均乘度矩阵的逆矩阵，其实就是对邻接矩阵做个归一化，重构节点的邻接关系**

  <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20221111113140326.png" alt="image-20221111113140326" style="zoom:40%;" />

- 以图神经网络多分类为例，基本的计算公式：**注意 $\hat{A}$ 是一直不变的，就是说邻接矩阵是一直不变的。现在看来，邻接矩阵只是刻画的图节点的连接关系，本质上也是输入的特征罢了。**

  <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20221111114141098.png" alt="image-20221111114141098" style="zoom:40%;" />

- 图神经网络层一般不能堆太多，这个层的多，是感受野的层数，还是处理特征的迭代次数，讲得有点混乱

  - 最后又说的是层数多了，特征比较发散

  <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20221111114707342.png" alt="image-20221111114707342" style="zoom:40%;" />

#### PyG官方文档

##### PyG Introduction

- (11/16)[`.contiguous()`的作用](https://www.bilibili.com/read/cv18151474/)

- 将无向图刻画为有向图

  <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20221116175043154.png" alt="image-20221116175043154" style="zoom:50%;" />

- Data数据包含了节点级、边级、图级三类属性；以PyG基准数据集为例，导入后得到dataset，它的每一个元素都是一张图，`len(dataset)` = 600，说明有600张图；所有图被分为6类，节点特征维度均为3；以第一张图为例，该图有37个节点，**无向边为168/2=84条**

  <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20221116181901448.png" alt="image-20221116181901448" style="zoom:67%;" />

- 用mask标记了该节点数据的用途(训练、验证or测试)

  <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20221116182037964.png" alt="image-20221116182037964" style="zoom:67%;" />

- 加上自连接后edge_index的维度由`(2,E)`变为了`(2,E+N)`

- 度矩阵维度 `(1,N)`

##### GCN从底层到实现

- (11/16)今天把这个专栏从头到尾的精华吸收了  <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20221115224604951.png" alt="image-20221115224604951" style="zoom:60%;" />

- 输入数据只要满足格式要求，不同层之间就只需要考虑数据的特征数

- [`model = model.to(device)` 原理](https://blog.csdn.net/weixin_44010756/article/details/115941131)

- [`model.train()、model.eval()` 原理](https://blog.csdn.net/Qy1997/article/details/106455717/)

- **以官方文档最简单的GCN model为例，再次梳理统一的步骤**

  1. 堆叠模型类，只需关注特征维度
  2. 导入数据，并指定设备`device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`，然后**将声明的模型和数据都加载到该设备上：`data.to(device)&model.to(device)`**
  3. 声明优化器
  4. `model.train`
  5. 迭代训练(语句顺序恒定)：`optimizer.zero_grad()` -> `out = model(data)`计算预测值 -> 计算损失值 -> `loss.backward()` -> `optimizer.step()`
  6. `model.eval`
  7. 计算精度

  ```python
  from torch_geometric.datasets import Planetoid
  import torch
  import torch.nn.functional as F
  from torch_geometric.nn import GCNConv
  
  class GCN(torch.nn.Module):
      def __init__(self):
          super().__init__()
          self.conv1 = GCNConv(dataset.num_node_features, 16)
          self.conv2 = GCNConv(16, dataset.num_classes)
  
      def forward(self, data):
          x, edge_index = data.x, data.edge_index
          x = self.conv1(x, edge_index)
          x = F.relu(x)
          x = F.dropout(x, training=self.training)
          x = self.conv2(x, edge_index)
          return F.log_softmax(x, dim=1)
  
  dataset = Planetoid(root='/tmp/Cora', name='Cora')
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = GCN().to(device)
  data = dataset[0].to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
  model.train()
  for epoch in range(200):
      optimizer.zero_grad()
      out = model(data)
      loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
      loss.backward()
      optimizer.step()
  model.eval()
  pred = model(data).argmax(dim=1)
  correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
  acc = int(correct) / int(data.test_mask.sum())
  print(f'Accuracy: {acc:.4f}')
  ```

- 如激活函数层，就只需要传入数据本身`x`，对于图神经网络层，就还需要传入邻接矩阵(即`edge_index`)，`x = self.conv1(x, edge_index)`

- PyG用到的Data一般都需要使用到mask

##### 消息传递机制

- 

##### 数据集制作

#### PyG分类实战

> 参考[CSDN PyG专栏](https://blog.csdn.net/cyril_ki/category_11595235.html)

### DGL

#### DGL图节点分类

> DGL入门、分类任务及其它任务均可参考[官方文档](https://docs.dgl.ai/tutorials/blitz/1_introduction.html)

- 流程

  1. 加载 DGL 提供的数据集
  2. 用 DGL 提供的神经网络模块构建 GNN 模型
  3. 在 CPU 或 GPU 上训练和评估用于节点分类的 GNN 模型

- 在图神经网络出现之前，许多提出的方法要么单独使用连接性（**例如 DeepWalk 或 node2vec**），要么使用连接性和节点自身特征的**简单组合**。相比之下，GNN 提供了通过结合局部邻域的连通性和特征来获得节点表示的机会，本质上就是节点自身特征和结构更复杂的组合

- 本教程将展示如何在 Cora 数据集上构建这样一个用于半监督节点分类的 GNN，该数据集是一个以**论文为节点**、**引文为边**的引文网络。任务是**预测给定论文的类别**。**每个论文节点都包含一个字数向量作为其特征**，**已归一化以便它们总和为 1**，和GCN原始论文过程一致

- [DGL 0.4.0版本没有Cora数据集，pip先卸载再安装指定的新版本，之后还需要下回来，因为CoLA模型只能用0.4.0版本](https://zhuanlan.zhihu.com/p/343956193)

  - `pip show dgl` 查看当前虚拟环境下载的包版本

  - `pip uninstall dgl` 卸载

  - `pip install dgl==0.5.3` 下一个稍微新一点的老版本，然后还是报了一个小错误，默认`pip install dgl`下载了0.9.1版本，就正常了

    <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20221214110916236.png" alt="image-20221214110916236" style="zoom:67%;" />

##### 加载数据集

- DGL数据格式

  <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20221214150640181.png" alt="image-20221214150640181" style="zoom:40%;" />

- `ndata`为节点属性，`edata`为边属性

##### 定义GCN

- 要构建多层GCN，直接堆叠`dgl.nn.GraphConv` 模块即可，这些模块继承自`torch.nn.Module`
- `g.ndata['feat'].shape[1]` 就是**初始的节点特征维度**

##### 训练GCN

- 步骤review

  1. 堆叠模型类，只需关注特征维度
  2. 导入数据，并指定设备`device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`，然后**将声明的模型和数据都加载到该设备上：`data.to(device)&model.to(device)`**
  3. 声明优化器 `optimizer`
  4. `model.train`
  5. 迭代训练(语句顺序恒定)：`optimizer.zero_grad()` -> `out = model(data)`计算预测值 -> 计算损失值 -> `loss.backward()` -> `optimizer.step()`
  6. `model.eval`
  7. 计算精度

- **前向传播深入理解**

  1. 在定义好GCN后，主函数中`model = GCN(g.ndata['feat'].shape[1], 16, dataset.num_classes)`传入具体的参数，进行模型的实际搭建；**所以只要将原始数据的特征维度和代码匹配，就可以正常输入输出了(already practiced)**

     <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20221214155305851.png" alt="image-20221214155305851" style="zoom:40%;" />

  2. 训练方法传入图和模型两个参数 `train(g, model)`，然后`features = g.ndata['feat']`，再将图和特征传入model：`logits = model(g, features)`，这个features是包含所有节点的特征矩阵，调用`model()`时自动调用定义的forward方法，参数也是契合的<img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20221214153647241.png" alt="image-20221214153647241" style="zoom:25%;" />，然后调用堆叠的模型层中，如`self.conv1 = GraphConv(in_feats, h_feats)`，**会自动只关注features的第二维**

  3. 通过神经网络层维度改变；通过激活函数维度不变，因为激活函数只是对向量中的每一个元素分别操作；**最终返回的本质上就是将特征压缩后的特征矩阵**

     ​							<img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20221214153941761.png" alt="image-20221214153941761" style="zoom:40%;" /> <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20221214154040160.png" alt="image-20221214154040160" style="zoom:50%;" />

























