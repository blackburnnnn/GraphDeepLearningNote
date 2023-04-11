## 当前数据集制作工作总结

### 步骤 

1. 恶意推广用户177个，集中于https://github.com/blackburnnnn/octo-meme仓库，正常用户全从https://github.com/AutumnWhj/ChatGPT-wechat-bot中获取
2. GitHub用户节点46维特征矩阵构建
   1. GitHub用户基本信息，共13维

      - 8个数值字段(.mat中都需保存为double类型)：`public_repos`、`owned_private_repos`、`public_gists`、`followers`、`following`、`collaborators`、`账号创建距今时长(单位年，2位小叔)`，`账户更新距今时长(单位年，2位小数)`

      - 5个非数值字段(非None为1，None为0)：`company`、`blog`、`email`、`location`、`bio`

   2. 基于X-Lab全域日志获取GitHub用户行为信息，共30维
      - 总共15种行为事件的总数
      - 选定GitHub比例最多的5种事件定义为显著：ForkEvent、IssueCommentEvent、IssueEvent、PullRequestEvent、WatchEvent；当前用户5种最显著事件发生间隔的平均值、日均次数、次数占其总行为次数的比例
3. GitHub用户邻接矩阵构建
   - 节点(用户)之间的关系基于5种显著行为 `marked_actions = ['ForkEvent', 'IssueCommentEvent', 'IssuesEvent', 'PullRequestEvent', 'WatchEvent']` + follow&following关系
   - 正常仓库5类行为的比例是18.5%，9.1%，3.9%，0.3%，78.8%，因为异常仓库中只有ForkEvent和WatchEvent，我自己进行了用户指定，形成了和正常仓库相近的比例：18.5%，9.2%，4%，1%，69.9%
   - 所有的连边是随机选择的，事件数量由高到底为WatchEvent，ForkEvent，IssueCommentEvent，IssuesEvent，PullRequestEvent，越少的事件价值越高，直观上他们的连接就更紧密，若该仓库涉及该事件的人数为k，则边的数量分别扩充为1k，1.25k，1.5k，1.75k，2k，边的权重也扩大为1，1.25，1.5，1.75，2
   - 存在follow或被follow关系的节点，连边的权值设置为2
   - 群体之间的关系构建
     - (异常row，正常col) 177个 value为异常群体内value均值
     - (正常row，异常col) 177个 value为正常群体内value均值

### 问题所在

1. 实际上异常仓库内的行为数据并不充分，只有star和fork

## Tools

- 程序运行耗时计算

  ```python
  from datetime import datetime
  starttime = datetime.now()
  endtime = datetime.now()
  delta = (endtime - starttime).microseconds/1000 # 单位ms
  ```

- [python `if __name__ == '__main__'` 作用](https://www.jb51.net/article/241090.htm)，其实只是在当前文件被import时起作用

### Pandas

- [dataframe条件筛选与修改](https://zhuanlan.zhihu.com/p/581436381)

## 数据集制作全流程

### 恶意推广用户获取

- 仓库链接 https://github.com/blackburnnnn/octo-meme
- 300元×1.35的友善套餐：100star + 32fork + 24watch + 20follow  ![image-20230310171337928](/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230310171337928.png)
  - 所有行为由不同用户操作
  - 40%以上的用户之间有follow关系

- 实际上买到的就只有star和fork

### 节点特征矩阵

- PyGitHub

  > 参考 [PyGitHub官方文档](https://pygithub.readthedocs.io/en/latest/index.html)、[PyGitHub官方仓库](https://github.com/PyGithub/PyGithub)、[郭飞的笔记](https://www.guofei.site/2019/11/17/pygithub.html)、[python技术站](https://thepythoncode.com/article/using-github-api-in-python)、[wangsong PyGitHub](https://github.com/wangshub/who_is_following/blob/master/github_followers_api.py)、[PyGitHub用户分析&图可视化](https://luzhijun.github.io/2016/10/02/GitHub%E6%95%B0%E6%8D%AE%E6%8F%90%E5%8F%96%E4%B8%8E%E5%88%86%E6%9E%90/)

  - username <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230228101720489.png" alt="image-20230228101720489" style="zoom:80%;" /> 密码照常小写


  - token `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`


  - 不需要考虑仓库具体内容，检索一下仓库-用户的从属关系即可，最主要的是需要挖掘follower和following以及用户个人信息


  - 经过调试，`g = Github()` 是无条件访问，会出现RateLimitExceededException，`g = Github(username, password)` 在访问具体信息如用户对应仓库时也会出现认证要求<img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230228224210327.png" alt="image-20230228224210327" style="zoom:60%;" />，`g = Github(token)` 是最好的，而且也远远超过了官方所说的每小时5000次请求(原因暂时不明)，之后就一直基于这个入口，参考PyGitHub官方文档去获取信息

    <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230228224316097.png" alt="image-20230228224316097" style="zoom:77%;" />


  - (3/8)从[pygithub1](https://vimsky.com/zh-tw/examples/detail/python-ex-github-Github---class.html)、[pygithub2](https://vimsky.com/zh-tw/examples/detail/python-ex-github-Github-get_followers-method.html)、[pygithub3](https://vimsky.com/zh-tw/examples/detail/python-ex-github-Github-get_following-method.html)，可以直接获取当前用户所有star过的repo列表，也可以直接获取到当前repo的fork列表


#### github用户基本特征选取

- 仓库只是作为支点，所以不用去管仓库Class拥有哪些属性，在GitHub层面只需要去管User的属性，比如company，location等，当然会存在很多空值；具体有哪些字段见[官方文档](https://docs.github.com/zh/rest/users/users?apiVersion=2022-11-28#get-a-user)，如下所示

  ```json
  {
    ...
    "company": "GitHub",
    "blog": "https://github.com/blog",
    "location": "San Francisco",
    "email": "octocat@github.com",
    "bio": "There once was...",
    "public_repos": 2,
    "public_gists": 1,
    "followers": 20,
    "following": 0,
    "created_at": "2008-01-14T04:33:35Z",
    "updated_at": "2008-01-14T04:33:35Z",
    "private_gists": 81,
    "total_private_repos": 100,
    "owned_private_repos": 100,
    "disk_usage": 10000,
    "collaborators": 8,
  }
  ```

- 甚至pycharm都可以直接显示可选字段 <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230308183343903.png" alt="image-20230308183343903" style="zoom:70%;" />

- 选取的个人信息字段

  1. 选取或构建8个数值字段(.mat中都需保存为double类型)：`public_repos`、`owned_private_repos`、`public_gists`、`followers`、`following`、`collaborators`、`账号创建距今时长(单位年，2位小叔)`，`账户更新距今时长(单位年，2位小数)`
  2. 选取5个非数值字段(非None为1，None为0)：`company`、`blog`、`email`、`location`、`bio`

#### X-lab全域数据集用户特征&选取

- X-lab全域数据集介绍，参考[官方github](https://github.com/X-lab2017/open-digger)

- [数据库字段文档](https://github.com/X-lab2017/open-digger/blob/master/docs/assets/data_description.csv)

- 基于全域数据集罗列所有包含的行为

- sql基础分析

  - 查看2022年单月在github有记录用户个数

    ```sql
    select count(distinct (actor_id)) from gh_events
    where created_at > '2022-04-15 03:59:55' and created_at < '2022-05-15 20:11:07'
    ```

  - clickhouse数据库 `actor_login`是用户名，`type`和`action`是事件，查询的时候限制一下时间范围，否则消耗时间太多

    <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230308231249545.png" alt="image-20230308231249545" style="zoom:80%;" />

    - 查询所有不同的事件类型，共15种 `select distinct type from gh_events`  <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230308231149480.png" alt="image-20230308231149480" style="zoom:90%;" />，其中CreateEvent、DeleteEvent、PushEvent等行为少数高级别开发者才有权限，显著性太高，不选择，**故初定选择五种关系**
      - **ForkEvent**
      - **IssueCommentEvent**
      - **IssueEvent**
      - **PullRequestEvent**
      - **WatchEvent**

  - 事件的触发方式类型有8种  <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230308231509239.png" alt="image-20230308231509239" style="zoom:80%;" />

  - 缩小查询的时间范围到2023年 `and created_at > '2023-01-01 00:00:01'`

  - **对于正常仓库，叉掉的5个行为事件数都很少，204仓库和微信chatgpt仓库事件总数和选取的5种事件类型的数目，以及5种事件分别对应的aciton类型**

    ![image-20230310175037405](/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230310175037405.png)

    - 事件总数 52：4929；
    - ForkEvent 7：697；都只有added
    - IssueCommentEvent  6：747；都只有created
    - IssuesEvent 5：286；opened、后者多了closed、reopened(比例为193：91：2)
    - PullRequestEvent 16：25；opened+closed(比例为8：8)、后者多了reopened(比例为14：10：1)
    - WatchEvent 2：3115；都只有started
  
  - 在SQL语句中，where子句并不是必须出现的，where子句是对检索记录中每一行记录的过滤，having子句出现在group by子句后面；在一句SQL语句中，如果where子句和group by……having子句同时都有的话，必须where子句在前，group by……having子句在后

#### 节点46维特征整合

- 特征矩阵每一行：**github基本用户信息+github全域数据库的个人行为**

  - 已经有了13个数值或非数值的基本用户信息特征，对于初步选定的5种行为，基于这两方面都还需要构建出合适的特征

  - 可参考之前的工作：构建了28维特征

    <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230310231843342.png" alt="image-20230310231843342" style="zoom:87%;" />

    - star和fork的总数 1
    - \~占所有操作数的比例 1
    - 不同操作的数量 14
    - 一分钟/一小时/一天内的均值、方差、变异系数 9
    - ...

- (3/12)对应的type和action：1、ForkEvent(added)；2、IssueCommentEvent(created)；3、IssuesEvent(opened)；4、PullRequestEvent(opened)；5、WatchEvent(started)

  - (3/13)不考虑action了，反正不会丢失太多数据

- 参考github推广文献对每个用户进行特征设计

  > 共43维特征，且不需要做太多的特征工程，之后就不用SVM这些模型去比，直接用图嵌入模型和图神经网络模型

  1. 当前用户8个数值型github用户基本属性

  2. 当前用户5个非数值型github用户基本属性

  3. 当前用户2023年以来15种行为事件的总数 <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230308231149480.png" alt="image-20230308231149480" style="zoom:90%;" />，以选择的正常仓库为例，23年以来这15种事件发生次数依次为

     - 1，0，14，0，13，480，0，460，156，0，0，0，0，2032,   2

     - Vscode15种事件的比例

       <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230405201326682.png" alt="image-20230405201326682" style="zoom:80%;" />
     
  4. 当前用户2023年以来5种最显著事件发生间隔的平均值(单位s)
  
  5. 当前用户2023年以来5种最显著事件发生的日均次数
  
  6. 当前用户2023年以来5种事件次数占其总行为次数的比例(小数)

- Pycharm配置clickhouse

  ```json
  remote_address(host) = "cc-uf6764sn662413tc9.public.clickhouse.ads.aliyuncs.com"  # 远程数据库服务器地址
  remote_user_name = 'xlab'  # 远程数据库服务器用户名
  remote_user_passward = 'Xlab2021!'  # 远程数据库服务器密码
  database = 'opensource'  # 远程数据库名
  ```

- [python连接clickhouse](https://blog.csdn.net/qq_45956730/article/details/127246423)，注意端口号应为9000

- sql格式和操作消耗时间

  ```python
  starttime = datetime.datetime.now()
  sql = """
      select * from gh_events
      where repo_name = 'blackburnnnn/octo-meme';
  """
  res = client.execute(sql)
  endtime = datetime.datetime.now()
  delta = endtime - starttime
  print(res)
  print(delta.microseconds/1000,'ms')
  ```

- (3/13)github用户基本信息特征已构件好，还有30维特征，对于5种显著行为，只考虑type不考虑action了

  ```python
  # 所有15种行为，前5种为显著行为
  all_actions = ['ForkEvent','IssueCommentEvent','IssuesEvent','PullRequestEvent','WatchEvent',
                 'CommitCommentEvent','CreateEvent','PushEvent','DeleteEvent','GollumEvent',
                 'MemberEvent','PublicEvent','PullRequestReviewCommentEvent','ReleaseEvent','PullRequestReviewEvent']
  ```

1. 15种行为次数：count_1~count_5
   - 次数返回res是[(8,)]格式， `res[0][0]`取int型次数
   - sql传参要记得加引号！ <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230313230804351.png" alt="image-20230313230804351" style="zoom:80%;" />
2. 5种显著行为发生次数占总行为次数的比例(小数)：ratio_1~ratio_5
3. 5种显著行为发生的日均次数：perd_1~perd_5
   - 按上述时间计算为74天
4. 5种显著行为发生的月均次数：perm_1~perm_5
   - 按上述时间计算为3个月
5. 5种显著行为发生的秒均次数：perm_1~perm_5
   - 和2、3不同，需要计算时间戳

- 共46维特征

#### 特征矩阵持久化

- (以CoLA代码为例)读取的.mat本身就是字典类型：{异常标签：{xxx}，特征矩阵：{xxx}，邻接矩阵：{xxx}}

  ![image-20230311224729086](/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230311224729086.png)

  1. 异常标签为ndarray类型，`np.array`创建

  2. 特征矩阵为csc_matrix类型，可以在ndarray的基础上再调用`sp.csc_matrix`进行创建，所以可以通过pygithub和python先用户基本信息和行为信息使用pandas持久化在csv中，再读取csv得到ndarray格式的数据，去除表头就可以直接调用`sp.csc_matrix`得到邻接矩阵了

     <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230311225033175.png" alt="image-20230311225033175" style="zoom:80%;" />

  3. 邻接矩阵为csc_matrix类型，在github用户场景下，行号和列号就是用户id，值就是两者之间的连接权重，如下图创建coo\_matrix后调用`.tocsc()`转化为csc_matrix即可

     - **之后还需要将用户id和用户名进行映射**

     <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230311225552896.png" alt="image-20230311225552896" style="zoom:77%;" />

- 储存为.mat `sio.savemat('./mat_demo.mat',{'Network': network, 'Label': label, 'Attributes': attr, 'Class': labels})`

  <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230311230443043.png" alt="image-20230311230443043" style="zoom:80%;" />

- 基于pandas设计csv结构 <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230313183743236.png" alt="image-20230313183743236" style="zoom:67%;" />

  1. 8个数值字段：`public_repos`、`owned_private_repos`、`public_gists`、`followers`、`following`、`collaborators`、`账号创建距今时长(单位年，2位小叔)`，`账户更新距今时长(单位年，2位小数)`
  2. 5个非数值字段(非None为1，None为0)：`company`、`blog`、`email`、`location`、`bio`
  3. 15种行为次数：count_1~count_5
  4. 5种显著行为发生间隔的平均值：interval_1~interval_5
  5. 5种显著行为发生的日均次数：perday_1~perday_5
  6. 5种显著行为发生次数占总行为次数的比例(小数)：ratio_1~ratio_5

- 得到特征矩阵csv![image-20230313192343624](/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230313192343624.png)后需要将其转化为.mat格式![image-20230313192319892](/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230313192319892.png)，注意.csv中index是0起，.mat中是1起，**其实就是把pandas读取csv获得的DataFrame类型的二维矩阵转化为csc_matrix类型的二维矩阵**

- **特征矩阵中所有数据要保证为float类型，在最初保存为.csv时就进行操作**

  <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230313195038013.png" alt="image-20230313195038013" style="zoom:80%;" />

- 且最初保存csv的时候要表头不要索引<img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230313195604151.png" alt="image-20230313195604151" style="zoom:67%;" />，会得到这样的csv<img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230313195714648.png" alt="image-20230313195714648" style="zoom:80%;" />，然后`pd.read_csv()`读取之后在控制台也会自动生成索引，维度也匹配(节点数，特征数)

  <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230313195804510.png" alt="image-20230313195804510" style="zoom:80%;" />

### 节点邻接矩阵

- 关系构建的2个来源：followers&following关系使用PyGitHub，用户之间的其他关系基于repo进行关联

- 调用sparse稀疏矩阵，要注意后面基于GCN的对称化和values的兼容

  <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230307230522021.png" alt="image-20230307230522021" style="zoom:70%;" />

- 需要构建出这样的索引关系，另外我这个是带权图，还需要自定义value，不应该像上图一样都为1

  ```)json
  (0, 0)
  (1, 3)
  (4, 8)
  ...
  ```

#### CoLA邻接矩阵条件

- 基于CoLA的邻接矩阵构建规则

  1. `values`、`rows`、`cols`、`label`、`labels`、特征矩阵`zz` 6个列表长度要一致
  2. 确保`rows`所有节点的id都存在，这样在邻接矩阵规范化时不会有除0警告
  3. 修改长度之后记得修改`sp.coo_matrix`中的`shape`

  <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230314190454054.png" alt="image-20230314190454054" style="zoom:80%;" />

  - 报了这个离奇的bilinear错，极其神奇，17个节点不行，16 18 19 20个都不报错

    <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230314191301416.png" alt="image-20230314191301416" style="zoom:67%;" />

- **只要行和不为0即可，这样就不会出现警告，而且这样在我的构建方法中，所有节点的都出现在了rows中，就算不连通也不会报错了！**

  ```python
  values = [9, 1, 1, 1, 2, 3, 0, 0, 6, 1, 3, 4, 5, 0, 4, 2, 1, 1, 99, 3, 3, 3, 3, 3, 3] # 终于成了！！！这里面不能有独立的节点！一个repo内肯定都有联系
  rows = [0, 3, 1, 1, 2, 3, 5, 5, 6, 1, 2, 4, 5, 5, 4, 6, 7, 7, 7, 8, 9, 10, 11, 12, 13]
  cols = [1, 2, 4, 5, 5, 4, 6, 7, 7, 0, 1, 1, 1, 2, 3, 5, 5, 6, 8, 7, 2, 3, 7, 7, 7]
  network = sp.coo_matrix((values, (rows, cols)), shape=[14, 14], dtype=float) # 貌似必须是4的倍数！！！艹 好像也不需要啊！！！之后再看！
  a = network.todense()
  print(a)
  print(a.shape)
  
  count1 = 0
  count2 = 0
  for i in range(a.shape[0]):
      if a[0].sum() == 0:
          count1 += 1
  for i in range(a.shape[1]):
      if a[:,i].sum() == 0:
          count2 += 1
  print('行和为0：', count1)
  print('列和为0：', count2)
  
  b = nx.from_scipy_sparse_matrix(network)
  print(b)
  print(nx.is_connected(b))
  ```

- **同时要保证rows和cols对应的不能是两个相同的id！** ![image-20230315121414974](/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230315121414974.png)

  - **为什么行和不为零还是会在rowsum那里出警告：同时还需要保证上图的values中没有0值！！！**

#### 具体构建方式

- 不同行为的用户是一个列表，所有用户的`actor_login`对应有顺序的id列表，如`[0,1,2,3]`，基于这个列表去构建邻接关系，但是不同的构建方式有不同的时间复杂度，如果按照最初的设想，对于一个长度为n的id列表，时间复杂度为n的n次方，太高了，所以邻接矩阵的构建还需要从时间消耗和边的数量等角度考虑
- **不可能按n的n次方去构建，反正是对index列表进行操作，所以可以按照相同的模式，在所有节点连通的条件下，在正负样本群内随机选择相同的比例进行边的连接，这样也不用去借助repo的creator了，因为这样随机性会大大减少**
- networkx+scipy+pandas联动，利用好networkx判断连通性的接口
- 先将2023年以来的所有用户节点进行某种方式的权重0.1的弱连通，然后根据5种行为+follow关系使用贪心算法去修改
- 按用户名列表来做，然后把user-user-value存进csv，最后构建的时候获取用户的索引即可

##### 5种显著行为信息人数

- 5种显著行为 `marked_actions = ['ForkEvent', 'IssueCommentEvent', 'IssuesEvent', 'PullRequestEvent', 'WatchEvent']` + follow&following关系
  - 正常仓库总人数2576，5类显著行为的人数分别为478，234，100，8，2031，占总人数的比例分别为18.5%，9.1%，3.9%，0.3%，78.8%
  - 异常仓库总人数129+44=173，5类显著行为的人数分别为32，0，0，0，98
  - **把follower和watch的44个人分给除了Fork的另外4种行为**，和正常仓库类似，分配完后是32，0+16，0+7，0+2，98+23=121占总人数的比例分别为18.5%，9.2%，4%，1%，69.9%
  - **邻接矩阵的权重设定：初始先随机生成n条边的弱连接，初始value为0.5，保证第一行为所有索引，第二行索引值不同，values不为零即可**
  - index列表需要权值对称
    - 由比例可以看出，事件数量由高到底为WatchEvent，ForkEvent，IssueCommentEvent，IssuesEvent，PullRequestEvent，越少的事件价值越高，直观上他们的连接就更紧密，所以对于5类事件的更小团体，每个团体节点数为k，其中边的数量分别扩充为1k，1.25k，1.5k，1.75k，2k，value也扩大为1，1.25，1.5，1.75，2
      1. ForkEvent
      2. IssueCommentEvent
      3. IssuesEvent
      4. PullRequestEvent
      5. WatchEvent
    - 若对于该群体内存在follow或被follow关系，连边的权值设置为2
    
  - 所有用户的列表，肯定有5类显著行为都没做过的，所以一开始的权重也没必要那么低，而且契合row列表，讲故事就说初始状态保证每个节点至少有一条边

- 首先需要构建一个所有用户和index的映射关系，然后构建完之后将row、col中的用户名和index映射

##### follow信息

- 只考虑follower和following数量位于(0,100)的用户，否则有些几千个关注和被关注的，计算量太大

- **先把user1-user2写入dataframe，然后去重得到最终的名单，然后基于此名单得到最终的邻接矩阵**

- 在调用PyGitHub时一定要进行异常处理，否则运行很久做无用功

- 基于最终的正常群体初始邻接矩阵，不用再一千一千地去构建follow关系 <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230318003618208.png" alt="image-20230318003618208" style="zoom:80%;" />


##### 最终构建结果

- 异常用户最终的邻接关系已构建

  - 里面没有无法发起特征请求的用户，这个在构建好特征工程后还需要检验
  - 去重后也是173个，直接获取这173个节点的特征矩阵
  - 也需要将特征矩阵的用户名进行id映射，同时检查特征是否存在None值以及值的类型：已检查

  - 异常用户内部的邻接矩阵和特征矩阵都已完成，还需要进行用户名和index的id映射，这个事情在连接正负群体后进行操作，因为所有的id是针对于所有正负样本
    - 把多的4个follower也重新安排了，异常群体的时间范围弄到3/17

- 同时将特征矩阵的第一列也映射为id，特征矩阵第一列已经是用户名了


##### 社区间邻接关系&标签构建

1. 特征矩阵&邻接矩阵合并
2. 2550个正常标签 0，177个异常标签 1
3. 用户名-整型index的映射，储存为final邻接矩阵.csv&final特征矩阵.csv
4. 社区间邻接关系构建
   - (异常row，正常col) 177个 value为异常群体内value均值
   - (正常row，异常col) 177个 value为正常群体内value均值
   - 本来6524条边，加上之后为6878条

### 群体邻接矩阵&特征构建流程归纳

1. **群体初始邻接矩阵.csv：包含初始的异常群体177个节点+正常群体2550个节点**
2. 基于群体初始邻接矩阵构建**群体follow&following.csv**
3. 基于5种行为信息+follow关系构建**群体最终邻接关系.csv**，同时检验一下
4. 基于群体初始邻接矩阵.csv**统一构建用户特征矩阵.csv**
5. 用户名-id映射&空值检验&邻接矩阵合并$\Rightarrow$final邻接矩阵+final特征矩阵

### 构建.mat&最终数据集信息

- 需要保存4种信息，Label随便建个2727维的 ![image-20230318225147583](/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230318225147583.png)
- 删除了final特征矩阵.csv的用户名首列
- 4类属性类型 ![image-20230318231330762](/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230318231330762.png)

- NetWorkX查看图属性
  1. node数：2550+177=2727个
  2. edge数：...

### NetworkX

- (3/12)我跑CoLA用的是networkx2.6.3，是很新的版本，其实CoLA只是用到了dgl老的采样方法，这里我也不用改

- CoLA中也是基于稀疏格式的邻接矩阵先构建出了networkx的图 <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230312231903600.png" alt="image-20230312231903600" style="zoom:67%;" />，之后对于得到的这个nx_graph，可以参考[子豪兄的networkx教程](https://www.bilibili.com/video/BV1kM41147zV/?spm_id_from=333.788&vd_source=726461adc26f0b0f56256c07f5a478dc)进行具体分析，当务之急还是先构建好自己的特征矩阵和邻接矩阵跑CoLA的代码

  <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230312232054091.png" alt="image-20230312232054091" style="zoom:67%;" />

- [networkx和PyG联动](https://blog.csdn.net/vincent_duan/article/details/121381227)

- [networkx基础使用](https://blog.csdn.net/m0_37427515/article/details/112296656)

- CoLA代码中的networkx版本有`from_scipy_sparse_matrix`方法，可以直接基于系数矩阵进行构建 构建出的图类型是`networkx.classes.graph.Graph`，和networkx3.0版本的输出类型一样

  <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230314171012507.png" alt="image-20230314171012507" style="zoom:80%;" /><img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230314171122645.png" alt="image-20230314171122645" style="zoom:80%;" />

- 这样构建的邻接矩阵是非对称但无向的，index多的话，是不会增加边的！

  <img src="/Users/leizhenhao/Library/Application Support/typora-user-images/image-20230316181558133.png" alt="image-20230316181558133" style="zoom:67%;" />
