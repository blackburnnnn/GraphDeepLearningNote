## 文献搜集

1. Understanding Promotion-as-a-Service on GitHub 推广用户分类
2. GitHub fake star检测 https://dagster.io/blog/fake-stars

## GitHub用户建模&异常检测文献阅读

1. Understanding Promotion-as-a-Service on GitHub：传统方法检测GitHub推广活动
   - 也研究的是用户层面，不过只构建用户特征，用传统方法SVM进行的分类，但是精度非常高99.1%
   - 将普通用户和推广用户进行划分的关键障碍
     1. 正常账户和异常账户行为都可以多样，也都可以缺乏多样性，可能非常相似
     2. 正常账户和异常账户都可能在短时间内产生大量行为
     3. 大多数推广账号为了伪装成普通账号逃避检测，都会对知名仓库进行star和fork操作
   - 收集了两种负样本（即正常账户）：流行的 GitHub 存储库的贡献者和在流行的存储库上提出有价值问题的人。选择了GitHub2018报告的前10种编程语言，并在https://gitstar-ranking.com/中为每种不同的语言找到了最受欢迎的存储库。从这些存储库中，我们收集了1,550 个用户作为负样本，其中包括 200 个一直非常活跃的高知名度用户和 1,350 个普通用户，至于怎么挑选的，论文没有细说，**但是说辞就是“挑选了”**
   - 特征的选择还是数量、操作间隔等；采用了6种传统的分类方法；然后对三类特征进行了有效性验证
   - 然后对识别出的可疑账户，进行了验证
2. [基于图神经网络算法的多分类水军特征建模及识别](https://kns.cnki.net/kcms2/article/abstract?v=3uoqIhG8C475KOm_zrgu4lQARvep2SAkueNJRSNVX-zc5TVHKmDNkvb3uCZwzo6ue0zpyZSJk3MLHx6o0pr-t7_Wi_Em2W_z&uniplatform=NZKPT)
3. Measuring user influence in GitHub: the million follower fallacy：用户影响力
4. Understanding the popular users: Following, affiliation influence and leadership on GitHub：用户影响力
5. User influence analysis for Github developer social networks：用户影响力
6. Bot Detection in GitHub Repositories：GitHub存储库
7. CNN-based malicious user detection in social networks：社交网络用户检测
   - 对用户基于兴趣进行用户画像，基于相似度进行分类

## 新思路

- (3/26)水军仍然会对正常的项目有行为 <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230326174605757.png" alt="image-20230326174605757" style="zoom:80%;" />，这是隐蔽自己的手段
- **模拟用户的行为进行场景建模**
  1. 一大堆人都在开发，旧仓库不断迭代，新仓库也不断产生
  2. 恶意推广用户的存在场景
     1. 旧仓库购买推广，无论是大公司以往的仓库(如百度)，还是小公司产品急需推广，它们都会购买
     2. 为了推广或者如蜜罐这样产生的新仓库
  3. 异常用户
- **标签定义**
  1. 购买服务的这些人就是异常的，标签少的情况下，就更要利用好这少量的标签
  2. 如毕所说，怎样去确定一个用户的类型，不能只按特征，否则还用邻接矩阵干嘛？
- **用户影响力**
- ljs的异常仓库标签的确定方式？
- 目前的177个异常用户，可以基于他们过往的日志获取其他的仓库信息

## 新思路需要说服他人的点

1. 基于搜集的真实数据，可否模拟他们的行为进行建模？
2. 别人只用SVM来处理用户特征，那我为什么要用邻接矩阵呢？（即为什么不用特征建模，而要用图神经网络建模？）
   - 使用图神经网络的方法可以更好地处理节点之间的复杂关系和动态变化，即考虑了节点之间的连接信息
   - 可以利用半监督或无监督学习方法来训练模型，减少对标注数据的需求
   - (3/28)从建模角度来说，我为什么要将这个问题建模在图神经网络上呢？
3. 用图神经网络那么特征怎么构建？
   - 自己定义节点自身的属性
4. 怎样确定用户的标签？
5. 怎样刻画用户之间的关系？即邻接矩阵
6. 不能太粗地仅选择一个正常仓库

## 目前针对于论文合理性的关键问题

1. Understanding Promotion-as-a-Service on GitHub 这篇论文研究问题和我一样，数据获取方法和我一样，特征处理和我差不多，传统方法分类精度99%，而且我用图神经网络方法去研究，效果肯定没这么好，还怎么做下去？
   - 可以添加用户影响力这些东西？
   - 在用户的关系上做文章？
   - 所以将正常/异常用户建模为图应该更加严谨才说得通
   - 比模型就不和传统方法比，优化数据集建模，效果达到70％足矣，所以就需要挖掘用户之间的邻接关系
2. 如果对一些顶会的模型加以修改，效果还要好，那为什么还发c级小论文和华师大毕业论文？
3. (3/30)star和fork两类有效行为，异常数据量很少，不专注于特征工程，舍弃如短时间内爆发的特性，专注于结构信息的构建，先基于异常注入的方式理解异常注入，另外边的权重也舍弃掉；但是如果去构建全连接的话，相当于就是一个参数了，这是可以的吗？还需要好好看看异常注入和其他节点异常检测的文献
4. 缝合的模型在别人的基础上只好了一丁点，那怎么写在论文里呢？和谁比呢？



















