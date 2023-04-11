- (12/5) `git clone --depth=1 {ssh link}`
- 同质图和异质图的区别：同质图中只有一种类型的节点和边(例如,只有朋友关系的社交网络)，网络结构较为简单；因此，同质图神经网络通常只需要聚合单一类型的邻居来更新节点的表示即可(例如,通过在朋友关系下的邻居来更新节点表示)；但真实世界中的图大部分都可以被自然地建模为异质图(多种类型的节点和边，如IMDB数据中包含三种类型的节点Actor、Movie和Director，两种类型的边Actor-Moive和Movie-Director)；异质图神经网络就包含多种类型的节点和丰富的语义信息

### 论文

#### Abstract

- 近期研究的缺陷

- 归纳本文的贡献
  1. 复用GCN
  2. 使用自编码器重建原始数据结构

#### 1 Introduction

- 解释了为什么要使用GCN
- 除了GCN的具体功能外，还指出GCN在异常检测领域还没应用过
- 把GCN当做encoder，然后使用其他decoder，利用编码-解码过程中出现的error发现异常节点
- **本文的contribution**
  1. 分析了传统机器学习方法(浅层异常检测方法)的局限性
  2. 提出了DOMINANT模型，基于编码-解码过程，通过结构(邻接矩阵)、属性(节点特征)两个方面的重构error发现异常节点
  3. 多个baseline，多个数据集

#### 2 Problem Definition

- 对于包含n个节点信息的矩阵X，基于n个节点的异常程度进行排序

#### 3 The Proposed Model - Dominant

- 除了deep autoencoder思想，已经有很多基于重建的异常检测方法，只是自动编码这个方法效果最好

- 欧氏范数=L2范数=开根号(元素差的平方和)

- 在属性图上进行编码，基于network sparsity, data nonlinearity, complex modality interactions三个需要解决的问题讲故事，引入为什么要使用GCN

- GCN的原理：GCN适用于属性图，基于最初的节点特征矩阵X和邻接矩阵A，进行不断迭代

- **GCN就是encoder**

- sigmoid取值范围：(0,1)

- **结构重建** <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20221208180121617.png" alt="image-20221208180121617" style="zoom:87%;" />

  - **为什么提出这个公式也没讲，反正就是去拼向量**
  - 用经过GCN的节点特征向量刻画邻接矩阵，其实挺有道理的

- **属性重建**

  - 基于GCN的迭代公式 <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20221208180903363.png" alt="image-20221208180903363" style="zoom:80%;" />，再引入一个权重矩阵$W_3$，<img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20221208180941935.png" alt="image-20221208180941935" style="zoom:80%;" />，使用经过编码得到的$Z$，

- 联合使用结构重建和属性重建的误差，构建该**自动编码器**的**目标函数**

  <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20221208183302343.png" alt="image-20221208183302343" style="zoom:80%;" />

  - 超参数 $\alpha$ 平衡两类误差的影响
  - **矩阵的F范数** <img src="https://img-blog.csdnimg.cn/20201204103720487.png" alt="在这里插入图片描述" style="zoom:80%;" />，所以F范数的平方就是矩阵所有元素的平方和

- 打分 <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20221208184200646.png" alt="image-20221208184200646" style="zoom:80%;" />

  - 例如要估计节点i的分数，使用 $\hat{A}$ 和 $\hat{Z}$ 的第i列即可

- 计算复杂度分析：GCN的复杂度与网络的边数成线性关系，**这里引用了GCN原始论文的复杂度分析**

#### 4 Experiments

- 三个真实数据集在之前论文里都有出现

  1. BlogCatalog：一个博客共享网站。BlogCatalog中的博主可以相互关注，形成一个社交网络。**用户与一个标签列表相关联，用于描述自己和他们的博客，这些标签被视为节点属性**
  2. Flickr：一个图像托管和共享网站。与BlogCatalog类似，用户可以相互关注并形成社交网络。**用户的节点属性由反映用户兴趣的指定标签定义**
     - 对于GitHub数据而言，用户的特征还来自于用户的行为
  3. ACM：是一个引文网络，每篇论文都被视为网络上的一个节点，链接是不同论文之间的引文关系。**每篇论文的属性都是从论文摘要中生成的**

- 这三个数据集本来没有异常，是基于之前的论文对其进行了结构异常注入和属性异常注入

  - 构造的结构异常和属性异常节点数目相同
  - 属性异常的注入比较创新

- **选择了5个之前的baseline进行分析**

  - 我应该如何去选？

- **评价指标**

  1. ROC-AUC：AUC是ROC曲线下方的面积，越接近于1说明性能越好
  2. Precision@K：在最异常的K个节点中，成功检测出异常的比例
  3. Recall@K：检测出的所有异常，占全部异常的比例

- **参数设置**

  - 可以这样数据造假吗？

  <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20221208230130888.png" alt="image-20221208230130888" style="zoom:77%;" />

- **实验结果**：分别使用图和表格进行结果展示，并得出性能方面传统模型<**残差分析网络(还不太懂)**<本文网络的结论，并分析了原因

  1. **ROC curves and AUC scores of all methods on different datasets**

     ![image-20221208230352907](/Users/leizhenhao/Library/Application Support/typora-user-images/image-20221208230352907.png)

  2. 另外两个参数在三个数据集、四个模型上的表现

     <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20221208230732260.png" alt="image-20221208230732260" style="zoom:67%;" />

  3. 参数分析：创新点之一也是提出平衡节点属性和节点连接的超参数$\alpha$，然后去调整这个超参数，然后绘图

     <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20221208230745765.png" alt="image-20221208230745765" style="zoom:80%;" />

  

#### 5 Related Work

- 从属性图异常检测和网络数据深度学习两方面扯，我知道怎么水文字了，相关工作直接改别人的综述

#### 6 Conclusion

1. 使用GCN进行建模
2. 使用自动编码器框架(本质也没啥难的)
3. 未来的工作

#### 7 Acknowledgements

- 没啥用

### 代码

> [Code](https://github.com/kaize0409/GCN_AnomalyDetection_pytorch)

- 执行命令：`python run.py`

#### pytorch常用方法

- [ .view()](https://blog.csdn.net/scut_salmon/article/details/82391320)
- [.detach()](https://blog.csdn.net/qq_31244453/article/details/112473947)

#### argparse

- SimGNN

  > 参考 [python argparse使用方法](https://zhuanlan.zhihu.com/p/539331146)

  - 执行代码都是通过命令行，程序会识别键入了命令行，并通过argparse包自动解析，如果键入了未被配置的参数，则会报错

    <img src="/Users/leizhenhao/Library/Containers/com.tencent.qq/Data/Library/Application Support/QQ/Users/625779422/QQ/Temp.db/071B8FCF-7A72-4F98-BFF8-AA0540CFCF2F.png" alt="071B8FCF-7A72-4F98-BFF8-AA0540CFCF2F" style="zoom:70%;" />

  - **`nargs='*'`**：表示参数可设置**零个**或**多个**

    **`nargs='+'`**：表示参数可设置**一个**或**多个**

    **`nargs='?'`**：表示参数可设置**零个**或**一个**

  - [.set_defaults方法的作用](http://www.manongjc.com/detail/30-dxlzzmjqoqvrikb.html)：不通过解析命令行就配置自定义参数

  - add参数时是`-`分隔：`parser.add_argument('--learning-rate', type=int, default=4)`，最后调用的时候是用下划线`_`：`e = args.learning_rate`

  - **获取命令行参数参数&解析**，使用python内置方法 `vars` 返回**argparse对象的属性和属性值的字典对象**，然后使用[通用的套路](https://blog.csdn.net/itnerd/article/details/101543712)将命令行的参数可视化，这个链接和SimGNN可视化参数的代码一模一样

- DOMINANT

  - 右键运行`xxx.py`或者`python xxx.py`时参数都是默认的，`python xxx.py --epoch 100`时指定epoch参数
  - 使用参数的方法：`args.{xxx}`
  - 标准的训练程序都是键入命令行

#### 设备配置

- Ubuntu上的torch是无法使用显卡的 <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20221210184031126.png" alt="image-20221210184031126" style="zoom:67%;" />

#### 环境配置&其他问题

- [Python “Non-ASCII character 'xe5' in file” 报错解决](https://blog.csdn.net/taowuhua0505/article/details/80279631)
- 其他baseline的运行程序里面没有，不过只要把输入数据准备好，换个模型直接跑应该就行了
- 之后还需要学习模型调优，如**网格搜索法**

#### .mat数据读取&处理

- [稀疏矩阵存储理解](https://stackoverflow.com/questions/52299420/scipy-csr-matrix-understand-indptr) <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20221214221955006.png" alt="image-20221214221955006" style="zoom:43%;" />

- mathworks512邮箱&密码首字母大写

  <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20221216103606448.png" alt="image-20221216103606448" style="zoom:70%;" />

### 文章&代码关键点

- 给原本没有异常的数据生成异常，参考的是8，31两篇文献，实际上8也是参考的31，而且参考了另一篇关键文献；**代码里是没有异常注入这一步的！**

<img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20221214190530363.png" alt="image-20221214190530363" style="zoom:70%;" />

- 模型搭建和训练过程完整梳理
  1. 基本定义：节点特征矩阵X，邻接矩阵A
  2. 模型的三个基本组件
     1. encoder：使用GCN，结合节点特征和邻接矩阵信息进行节点嵌入表示学习
     2. structure reconstruction decoder 用学习到的节点嵌入重建网络拓扑
     3. attribute reconstruction decoder 用学习到的节点嵌入重建节点特征
  3. encoder
     - 整体思想：基于原始的输入，构建编码器和解码器，最小化这个目标函数 <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20221216171814754.png" alt="image-20221216171814754" style="zoom:67%;" />
     - 使用GCN结合X和A进行编码，得到包含属性和结构(A)信息的新节点特征矩阵Z
     - (12/27)平时有监督的学习过程，就是 $\min[\operatorname{dist}(\mathbf{X}, \operatorname{Model}(\mathbf{X})]$，而无监督的学习过程就是上面这样
     - **图神经网络层除了节点特征矩阵，还必须要输入邻接矩阵，否则无法进行邻居信息的聚合**
  4. structure reconstruction decoder
     - 对于某个节点，如果其结构信息可以通过结构重建解码器来近似，则异常的可能性很小。相反，如果不能很好地重建连接模式，则意味着其结构信息不符合大多数正常节点的模式
     - **重构得到一个新的邻接矩阵，即重建了结构，预测了i和j之间有没有链接** <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20221216172625956.png" alt="image-20221216172625956" style="zoom:80%;" />
     - 结构重建误差 <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20221216175601299.png" alt="image-20221216175601299" style="zoom: 67%;" />
  5. attribute reconstruction decoder
     - **使用Z和最初的邻接矩阵，得到重构的节点特征矩阵 $\hat{X}$**
     - 属性重建误差 <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20221216175616159.png" alt="image-20221216175616159" style="zoom:67%;" />
  6. 总体目标函数 <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20221216175814838.png" alt="image-20221216175814838" style="zoom:77%;" />，$\alpha$ 是平衡两类误差影响的超参数，训练完了之后，输入每个节点，从而得到每个节点的异常程度 <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20221216180340855.png" alt="image-20221216180340855" style="zoom:80%;" />，其中$\hat{A}$和$\hat{X}$是训练完成的
  7. 在数据集上进行异常注入，是为了后续让所有模型在上面进行统一评估
