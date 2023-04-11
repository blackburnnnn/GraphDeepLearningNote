### 论文

#### 引言&相关工作

- 异常的标签成本过高，以无监督方式进行
  - Deep Anomaly Detection on Attributed Networks
  - Scalable Anomaly Ranking of Attributed Neighborhoods
  - GitHub恶意推广用户标签获取成本过高，在这个条件下以新的模式建模GitHub用户数据

- 以往的图异常检测模型：AMEN、Radar、ANOMALOUS、DOMINANT、SpecAE

- 对比学习以往文献

  - A. v. d. Oord, Y. Li, and O. Vinyals, "Representation learning with contrastive predictive coding," arXiv preprint arXiv:1807.03748, 2018. 
  - T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, "A simple framework for contrastive learning of visual representations," arXiv preprint arXiv:2002.05709, 2020. 
  - R. D. Hjelm, A. Fedorov, S. Lavoie-Marchildon, K. Grewal, P. Bachman, A. Trischler, and Y. Bengio, "Learning deep representations by mutual information estimation and maximization," in International Conference on Learning Representations, 2018.

- 目前的图自动编码器 DOMINANT、SpecAE 的弊端

  - 不检测异常本身，只是简单地重建数据，重建是一种朴素的无监督学习方案，无法充分利用数据
  - 需要输入全图数据，对内存要求过大

- 结构异常（邻接矩阵）和上下文异常（特征矩阵）

  <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230410111113227.png" alt="image-20230410111113227" style="zoom:40%;" />

- 对比自监督也属于无监督

  - 主要研究实例对的匹配
  - 对比学习模型提供了一个特定的预测分数来衡量每个实例对中元素之间的一致性，并且该比例与实例的异常情况高度相关，因此对比学习模型的预测分数可以直接用于异常检测

- 模型的学习目标：区分实例对中元素之间的一致性，结果可进一步用于评估节点的异常情况

- 不需要在全网络上运行图卷积，成功地避免了内存爆炸问题

- CoLA是第一个基于对比自监督学习的图节点异常检测方法

- 网络嵌入&图神经网络，诸多参考文献，[两者的关系很类似](https://www.jianshu.com/p/4186e42199d0)

- CoLA的GNN模块可以选择包含GCN在内的任何GNN模型

- 

  

#### 问题定义

- 属性图包含邻接矩阵&节点特征矩阵 G = (A, X)
- **和DOMINANT一样，训练过程是无监督的，不利用标签信息，只在评估阶段使用**

#### 方法

1. **Contrastive Instance Pair Definition，采样过程虽然属于数据预处理的一部分，但也是其重要的创新点**，包含以下4个步骤

   1. Target node selection：无放回随机抽样
   2. Subgraph sampling：简言之就是选择节点附近的节点
   3. Anonymization(匿名化)：将初始节点的属性向量设置为0向量
   4. Combining into instance pair <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20221217171910189.png" alt="image-20221217171910189" style="zoom:70%;" />
      - **注意看Fig.3.** <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20221217180719011.png" alt="image-20221217180719011" style="zoom:70%;" />

2. GNN-based **Contrastive Learning** Model

   - 输入模型的都是实例对，**符号定义如下**，**标签也是人为规定的，训练目标就是让相似度$s_i$和$y_i$更接近**

     <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20221217172114591.png" alt="image-20221217172114591" style="zoom:80%;" />

     <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20221217172134512.png" alt="image-20221217172134512" style="zoom:80%;" />

   - 模型的三个组件

     1. **GNN module** <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20221217172955941.png" alt="image-20221217172955941" style="zoom:80%;" />
        - 输入是**属于局部子图的节点的特征矩阵**以及**此局部子图的邻接矩阵**
        - 除了局部子图，还要用DNN把target node映射到相同的嵌入空间，使其可比
     2. **readout(读出) module**
        - 把局部子图转换后的特征矩阵转换为一个向量(和target node的嵌入同维)
     3. **discriminator(鉴别) module**
        - 对比以上两个向量给出分数 <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20221217174031643.png" alt="image-20221217174031643" style="zoom:80%;" />
        - **这样就预测出了这个实例对的标签，每个实例对其实就包含一组节点-子图**
        - **本质就是通过节点和其子图(周围节点)的比较，来判定他是否异常**
     4. 以上三个组件就可以称为CLM，本质就转化为了一个二分类问题，使用BCE作为损失函数，小改了一下

3. Anomaly Score Computation

   - **除了和自己周围的节点比，还和其他的子图组成实例对去比较**
   - **正常节点**正对得分接近1，负对得分接近0，大致就是相似度得分；**异常节点**正对得分接近0.5
   - 进行**多轮采样**，以减小特殊情况的影响，轮数R设置为了超参数

4. CoLA的整体过程

   1. 在每个epoch中，**对每个node**生成一个正对和一个负对，**将正对负对得分直接扔进BCE损失函数，不利用标签直接进行无监督学习**
   2. 训练完成后**对测试集每个节点进行多轮采样**，然后获得异常分数
   3. **不需要将整个图输入到模型中，只需要输入实例对，从而减小了计算开销**

#### 实验

1. Datasets

   - 前两个是社交网络数据，后面的都是引文网络数据

     <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20221217185202918.png" alt="image-20221217185202918" style="zoom: 67%;" />、

   - 将结构异常和上下文异常注入到干净的原始数据集中，DOMINANT也参考了对应文献，之后需要好好看看；同时采取手段平衡两类异常的注入比例

   - 所有类别标签在训练过程中都被隐藏，只在推理阶段使用

2. Experimental Settings

   - Baselines：5个
   - Evaluation metrics：ROC-AUC曲线
   - Parameter Settings
   - Computing Infrastructures

3. Anomaly Detection Results

   - **创新点：DGI用的实例对是full graph v.s. node，本文用的实例对是target node v.s. local subgraph**
   - CoLA的空间复杂度与节点数无关

4. Parameter Study

   - Effect of the number of sampling rounds 𝑅
   - Effect of subgraph size 𝑐
   - Effect of embedding dimension 𝑑

5. Ablation Study

   - 这里的消融实验其实也是对比实验，比如把聚合方式由mean换成max

#### 总结

- 这篇文章的图数据结构不算异质图，从使用的数据集就可以明白

### 数据集异常注入

- 

### 代码

- 就两行命令，`python inject_anomaly.py --dataset cora` 处理原始数据集，`python run.py --dataset cora` 训练和推理

#### 1.参数&随机种子设置

- pycharm配置

  - 注释颜色 E74169，在def下方第一行三个双引号的注释自动为红色，这也是python方法注释的标准

    <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230115221754013.png" alt="image-20230115221754013" style="zoom:80%;" />

  - 关闭各种无意义提示

    - `typo:in word` 没有以驼峰形式命名
    - [`Cannot find reference ‘XXX‘ in ‘_init_.py‘`解决  ](https://zhuanlan.zhihu.com/p/143544588)，直接关掉也问题不大

- [随机种子](https://blog.csdn.net/qq_35568823/article/details/126461753)：包含4种基本随机种子和1个dgl随机种子，dgl设置随机种子就是为了在RWR随机过程保证一致性  <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230115102747604.png" alt="image-20230115102747604" style="zoom:80%;" />

- 参数均契合论文

  <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230114121402890.png" alt="image-20230114121402890" style="zoom:60%;" />    <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230114121416936.png" alt="image-20230114121416936" style="zoom:67%;" />     

- **运行配置**

  - torch 1.4.0

  - dgl 0.3.1：使用集成于其中的子图采样算法，高版本里面大概率是把这个方法淘汰了

  - sklearn：计算ROC和AUC

  - 硬件：Ubuntu 16.04、NVIDIA GeForce RTX 2070 (8GB memory) GPU、Intel Core i7-7700k (4.20 GHz) CPU and 15.6 GB of RAM(Random Access Memory，也叫主存，就是通俗所说的内存)

    > [参考查看当前Ubuntu配置](http://www.nndssk.com/xtwt/204689TGjcRp.html)

    - 内存大小：15.6GB  <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230114124902365.png" alt="image-20230114124902365" style="zoom:75%;" />

    - [`df -h`命令查看文件系统硬盘使用情况](http://c.biancheng.net/view/883.html)

      <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230115100824242.png" alt="image-20230115100824242" style="zoom:75%;" />

    - 处理器：Intel Xeon Platinum 8269CY(3.10GHz) CPU

#### 2.数据读取&预处理

- 经过异常注入后(不用搞)保存在dataset中，run.py也直接使用其中的数据，e.g.dataset中处理完成的cora数据如下

  - 基本介绍 <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230115230344882.png" alt="image-20230115230344882" style="zoom:90%;" />

  - .mat可视化

    ![image-20230115230521899](/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230115230521899.png)

- `scipy.io.loadmat` 读取.mat数据，在构建数据集的时候，就已经将数据类型和内容都用`sio.savemat`存储在.mat中，所以读出来就是对应格式，**.mat都是以字典格式进行存储的**

  <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230115233527928.png" alt="image-20230115233527928" style="zoom:100%;" />

- `def dense_to_one_hot(labels_dense, num_classes)`，传入类别向量`labels_dense` [2,3,4-1,0,2]和`num_classes` 6，**将类别向量[2,3,4-1,0,2]转化为one-hot矩阵，[这个代码是网上大家都在用的轮子](https://www.jianshu.com/p/e2fc6b7e00cd)，也可以直接使用sklearn中的OneHotEncoder实现** 

  ```python
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes  # [0, 6, 12, 18...2708*6]
  labels_one_hot = np.zeros((num_labels, num_classes)) # (2708, 6) 将类别标签映射为one-hot形式
  labels_one_hot.flat[index_offset+labels_dense.ravel()] = 1
  return labels_one_hot
  ```

  - 之前代码是不对的，修改了-1错误之后，AUC还上升了  <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230116235451426.png" alt="image-20230116235451426" style="zoom:60%;" />
  - Class在后面代码里是没有用到的

- [数据预处理部分使用到的轮子](https://blog.csdn.net/yihanyifan/article/details/125951572)

- 异常注入后的异常标签(1)是很稀少的

- **networkx和dgl搭配使用**

  - csr_matrix通过nx，变成nx的Graph，nx的Graph可以被输入至dgl，然后变为DGLGraph，这两种Graph都基于初始邻接矩阵整合节点信息和边信息

    <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230118121247351.png" alt="image-20230118121247351" style="zoom:80%;" />

- 数据预处理前后类型和shape的变化

  - 处理前，features：matrix(2708,1433)；adj：matrix(2708,2708)；labels：ndarray(2708,7)；idx_train&val&test:list；

    <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230211111409630.png" alt="image-20230211111409630" style="zoom:50%;" />

  - 处理后，features：Tensor(1,2708,1433)；adj：Tensor(1,2708,2708)；labels：Tensor(1,2708,7)；idx_train&val&test:Tensor(n,)

  - `[np.newaxis]` 直接在最外侧(左侧)加一个中括号，方便后续进行操作

    ```python
    x = np.matrix([[1,2,3], [4,5,6]])=>(2,3)
    y = torch.FloatTensor(x[np.newaxis])=>torch.Size([1, 2, 3])
    ```

#### 3.初始化模型&优化器

- [PyTorch self.modules()：遍历已搭建模型的各层](https://blog.csdn.net/weixin_42393848/article/details/120972653)
- 训练&推理进度条 <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20221218111735807.png" alt="image-20221218111735807" style="zoom:50%;" />

#### 4.子图采样

- 初始节点设置为目标节点，然后采样子图由目标节点的邻居节点组成。对于负实例对，初始节点是从除目标节点之外的所有节点中随机选择的。结果，目标节点和负对中的局部子图之间存在不匹配。

- 0.4版本dgl采样方法的文档都没了

- 基于之前的DGLGraph，返回的subgraphs嵌套列表是2708个子图id列表，每个列表包含4个节点id

  <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230122132021815.png" alt="image-20230122132021815" style="zoom:50%;" />

- 对于这个图结构，采样也没什么道理

  <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230224131259385.png" alt="image-20230224131259385" style="zoom:10%;" />

  ![image-20230225103205948](/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230225103205948.png)

- [networkx基础](https://www.bilibili.com/video/BV1kM41147zV/?spm_id_from=333.1007.tianma.1-1-1.click)

#### 5.训练

- `for epoch in range(args.num_epoch)` 迭代epoch，每个epoch都要训练所有数据，**在每个epoch开始时进行 `model.train()`**
- 在每个epoch中，将数据分为多个batch，`for batch_idx in range(batch_num)`，**在每个batch开始时进行 `optimiser.zero_grad()`**
- 损失函数 **`b_xent = nn.BCEWithLogitsLoss(...`**  <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230213224225610.png" alt="image-20230213224225610" style="zoom:80%;" />
  - bf和ba通过model得到预测的标签，然后直接调用`b_xent`获取总的损失
  - 如果设定`b_xent`为CrossEntropyLoss的话，损失值将为0，然后报错
- `added_adj_zero_col[:, -1, :] = 1.  # 每一个矩阵中的最后一行元素全为1(张量从左到右依次为哪个矩阵->矩阵的行->矩阵的列)`
- 

#### 6.推理

- **节点本身的类别标签完全未使用，全流程只用到了是否异常的标签，且在训练阶段不使用标签，推理阶段才用到**

### 结合论文看代码

- 构建target node v.s. local subgraph的实例对，使用readout模块提取目标节点和局部子图的低维嵌入

- 使用discriminator鉴别每个实例对的鉴别分数

- CoLA的三部分：定义对比实例对、基于GNN的对比学习模型、异常分数计算

#### A.对比实例对定义

- 以前也有提出实例对，但是都不是为异常检测任务提出的，这样做的动机就是因为异常通常反映在节点与其邻居的不一致上，与全局的信息无关，这样的设计可以专注于学习节点-局部子图的匹配模式来找出这种不匹配
- 采样流程
  1. 随机选择target node
  2. 基于1中选择的node，使用RWR对其周围的节点进行采样，代码中反映为上述的2708个子图id列表
  3. 匿名化处理：将局部子图中初始节点设置为0向量，避免对比学习模型轻易识别出局部子图中目标节点的存在
  4. 将目标节点和相关子图组合成实例对，保存进样本池中

#### B.GNN-based 对比学习模型

- 问题定义前面已经写过了，这部分包含3个模块：GNN module，readout module，discriminator module

1. **GNN module**

   - **目的是聚合当前局部子图节点之间的信息，并将高维属性转移到低维嵌入空间中**  <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230214122653165.png" alt="image-20230214122653165" style="zoom:80%;" />

   - 实际上就是这个阶段先把节点和子图节点的特征进行聚合转化，这个层也可以换成各种主流的GNN模型

     <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230214122811744.png" alt="image-20230214122811744" style="zoom:80%;" />

   - 因为是对比学习过程，局部子图是一个图，有邻接矩阵等结构信息，可以直接喂入GCN，但是与其相关的目标节点是一个没有结构信息的单独向量，所以需要使用之前的GCN相关的权重参数对其进行转化，视为最基本的DNN层

2. **readout module**

   - 目的是将GNN module转化得到的局部子图(包含目标节点)的多个向量，**整合为一个表达当前局部子图的单独向量**，这里就直接的average pooling function

     <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230214124140659.png" alt="image-20230214124140659" style="zoom:75%;" />	<img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230214124322263.png" alt="image-20230214124322263" style="zoom:75%;" />

3. **discriminator module**

### demo测试

#### 迷你数据集构建

- 构建8个节点的demo图，构建的时候不将对角线设置为1，预处理的时候会调用方法实现

- (2/20)需要的数据集属性(节点数为n，每个节点特征数为k) ![image-20230220184937987](/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230220184937987.png)

  - 邻接矩阵：`(n, n)`，对角线不为1，无向图，.mat中名称为`Network`，储存为`csc_matrix`类型
  - 节点特征矩阵：`(n, k)`，.mat中名称为`Attributes`，储存为`csc_matrix`类型
  - 是否异常标签，.mat中名称为`Label`，储存为`ndarray`类型

- [Python创建稀疏矩阵](https://blog.csdn.net/vor234/article/details/124935384)，下面的邻接矩阵构建就是参考的这个文档

- [scipy稀疏矩阵的创建tip与互相转化](https://blog.csdn.net/qq_36159768/article/details/108954721)

- [e.g.用户-商品矩阵转化及稀疏矩阵CSC，CSR，COO三种形式生成](https://blog.csdn.net/weixin_44731100/article/details/90244451)

- 搭建流程

  1. 先使用coo_matrix快速构建稀疏矩阵，然后调用to_csr()、to_csc()、to_dense()把它转换成CSR或稠密矩阵，所以之后基于GitHub用户特征构建的特征矩阵和邻接矩阵都可以按这个方式构建，注意有9×2个值

     ```python
     values = [1] * 18
     rows = [0, 1, 1, 1, 2, 3, 5, 5, 6, 1, 2, 4, 5, 5, 4, 6, 7, 7]
     cols = [1, 2, 4, 5, 5, 4, 6, 7, 7, 0, 1, 1, 1, 2, 3, 5, 5, 6]
     A = sp.coo_matrix((values, (rows, cols)), shape=[8, 8])
     print(A.todense())
     ```

     - `todense()`转化输出的稠密矩阵是`numpy.matrix`类型

     - 构建8个节点的迷你图，邻接矩阵和结构如下

       ​									<img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230224130734102.png" alt="image-20230224130734102" style="zoom:100%;" />  		<img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230224131259385.png" alt="image-20230224131259385" style="zoom:10%;" />

  2. 每个节点假设有3维特征，构建8×3的节点特征矩阵

     ```python
     zz = np.array([[1, 2, 0],
                   [2, 1, 1],
                   [1, 2, 5],
                   [7, 3, 3],
                   [1, 1, 1],
                   [0, 0, 1],
                   [0, 0, 2],
                   [1, 0, 3]], dtype=float)
     attr = sp.csc_matrix(zz)  # 邻接矩阵
     ```

  3. 异常标签：前4个正常(0)，后4个异常(1)

     ```python
     label = np.array([[1], [1], [1], [1], [0], [0], [0], [0]])  # 是否异常标签
     ```

#### 基础模型搭建

- 发现了[CoLA抄的代码](https://juejin.cn/post/7044873867252203557)











CoLA代码

1. 特征矩阵预处理

   - 测试数据

     <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230314231630485.png" alt="image-20230314231630485" style="zoom:70%;" />

   - 这里的预处理就是把每个节点的每一维特征除以行和

     <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230314233554992.png" alt="image-20230314233554992" style="zoom:70%;" />

2. 邻接矩阵预处理

   - 经历了这两步 <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230314234713823.png" alt="image-20230314234713823" style="zoom:70%;" />

   - 第一步就是按照GCN公式进行处理原始邻接矩阵->经过第一步->经过第二步的变化如下

     <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230314234750920.png" alt="image-20230314234750920" style="zoom:70%;" />

3. 模型架构&前向传播过程















