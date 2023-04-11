## Extra

### ICANN账户

- 一作是我，通讯作者是蒲老师，邮箱 ppu@cc.ecnu.edu.cn
- 研究生邮箱别名 zhlei@ecnu.edu.cn
- overleaf账号：研究生邮箱
- [icann注册流程](https://e-nns.org/icann2023/submission/)
  - 先用512邮箱创建账户(虽然已经改了zhlei的别名，但是还没生效)，老密码注册EasyAcademia，然后就可以登录icann投稿系统了

### Tools

- 翻译：DeepL

- 润色：火龙果
- 公式：Mathpix
- 表格：[Table辅助工具](https://www.tablesgenerator.com/#)，Table不同于图片，是用latex代码表达的
- 模型架构图绘制&训练截过图绘制
  - 需要转.eps格式
  - [论文绘图学习](https://www.bilibili.com/video/BV1z5411H72x/?spm_id_from=333.337.search-card.all.click&vd_source=726461adc26f0b0f56256c07f5a478dc)
  - [overleaf插入.eps格式图片](https://blog.csdn.net/qq_22812319/article/details/51889973)

### 投递

- 查重工具 crosscheck

- 导出所有的源文件

  <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230306105817499.png" alt="image-20230306105817499" style="zoom:75%;" />

- [基于overleaf论文投递技巧](https://blog.csdn.net/xovee/article/details/106250667)

- 参考文献需要由Bib转Bbl，[参考](https://blog.csdn.net/ing__/article/details/121711629?spm=1001.2101.3001.6650.9&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-9-121711629-blog-122350597.pc_relevant_multi_platform_whitelistv4&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-9-121711629-blog-122350597.pc_relevant_multi_platform_whitelistv4&utm_relevant_index=13)

### LaTex

- 逗号、句号、分号等标点后面要加空格，否则间距和换行都不正确

- [overleaf将参考文献格式bib转bbl](https://blog.csdn.net/ing__/article/details/121711629)，同时参考[此处](https://blog.csdn.net/qq_45027465/article/details/127737685)

- 将URL放入BibTeX文件中的步骤如下：

  1. 打开一个文本编辑器，例如记事本（Windows）或TextEdit（Mac）。

  2. 在文本编辑器中创建一个新文件，并将其保存为 .bib 扩展名的BibTeX文件。例如，可以将文件保存为“example.bib”。

  3. 在文本编辑器中添加一个新条目，格式如下：

     ```TEX
     @misc{<cite-key>,
       author = {<author>},
       title = {<title>},
       howpublished = {\url{<url>}},
       year = {<year>}
     }
     ```

     其中，`<cite-key>`是你为条目选择的引用键，`<author>`是文献的作者（如果有的话），`<title>`是文献的标题，`<url>`是文献的URL，`<year>`是文献的出版年份或更新日期。

     以下是一条示例URL条目：

     ```TEX
     @misc{google,
       author = {Google},
       title = {Google Search},
       howpublished = {\url{https://www.google.com}},
       year = {2021}
     }
     ```

  4. 保存并关闭BibTeX文件。

  5. 现在，你可以在LaTeX文档中引用这个BibTeX文件中的URL条目。在LaTeX文档中，你可以使用`\cite{<cite-key>}`命令来引用该条目。例如，`\cite{google}`将在文档中生成一个引用，指向名为“google”的URL条目。

  6. 请注意，BibTeX文件中的每个条目都需要一个唯一的引用键，以便在LaTeX文档中引用。此外，有些BibTeX风格可能不支持URL条目类型，因此请确保使用适当的风格文件。
  
- 插入公式

  - 

## 按序引用文献

1. GitHub维基百科 https://en.wikipedia.org/wiki/GitHub
2. GitHub fake star检测 https://dagster.io/blog/fake-stars
3. GitHub Promotion论文
4. GitHub REST API
5. X-lab GitHub全域数据库

## 具体写作

### 第一部分

#### 标题&摘要&关键字

1. 标题：Graph Contrastive Learning for Malicious Promoter Detection on GitHub
2. 作者
3. 摘要
4. 关键字：GitHub · Graph Contrastive Learning · Graph Neural Network · Malicious Promoter Detection

####  1 Introduction

> 文献引用：1.2.3.4.5

1. GitHub社区简介，社交媒体，GitHub恶意推广用户入侵问题，还没有工作对这个问题进行建模
2. 恶意推广者的存在&危害，检测的必要性，而且隐藏得和正常用户很相似
3. 目前方法的弊端，基于特征的方法不行，因此在特征工程的基础上基于真实数据构建了包含恶意用户的图网络，进行了适合GitHub用户的图嵌入
4. 同时为了提高训练的计算效率，引入了对比自监督的学习方法，构建了对比实例对，将图的全量数据分批进行训练，并提出了新颖的图神经网络架构
5. 最终实验结果表明，在恶意推广者的检测上，优于最先进的模型
6. Contribution
   1. GitHub数据集构建
      - 节点特征工程
      - 基于真实数据构建包含正常用户和异常用户的GitHub社区模型
   2. 使用对比自监督学习对GitHub用户进行采样
   3. 构建适合GitHub用户群体结构和特征的图神经网络模型
   4. GitHub用户预测&对比自监督学习效率分析

#### 2 Related Work

> 文献引用：

1. GitHub异常用户检测、社交网络恶意用户检测
   1. 
2. 图节点异常检测
   1. 
3. 对比有监督学习，以及与对比自监督的区别
   1. 文献参考CoLA和对比学习综述

### 第二部分

#### 3 Model Framework

- GCLProD

### 第三部分

#### 4 Experiments

##### Dataset Construction

- 

#### 5 Conclusion

### 图表制作&公式

1. 对比实例采样图
2. GNN模块图
3. 对比实验结果表格
4. 公式

























​	

