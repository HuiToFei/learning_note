# Python Notes

**Table Of Contents**

- [Pytorch](#pytorch)
  * [dataset](#Dataset)
- [callable_objcet](#2. callable-object)
- [Numpy](#3.numpy)
- [KNN](#knn)
  - [knn python](#knn-python)

### 1.Pytorch

#### 1.1 Dataset

Description: pytorch框架下数据集的准备

Class list: torch.utils.data.Dataset, torch.utils.data.DataLoader

Key Class: **torch.utils.data.DataLoader**

1) 常见数据集准备方式

编写自定义数据集,需要覆盖`__init__`, `__len__`,`__getitem__`三个函数，len函数用来返回数据集的大小

继承Dataset类

```python
from torch.utils.data import Dataset,DataLoader
import os 
import numpy as np

class MyOwnDataset(Dataset):
    def __init__(self,data_dir,transform=None):
        file_list = np.sort(os.listdir(data_dir))
        self.train_list=[os.path.join(data_dir,f) for f in file_list]
        
	def __len__(self):
        return len(self.train_list)
    
    def __getitem__(self,idx):
        data,label = np.load(self.train_list[idx])
        if self.transform:
            data,label=self.transform([data,label])
        return data,label

```

那么此时如果不对原始数据集进行任何其他的处理，例如shuffle,裁剪，压缩等等，网络训练就可以使用下面例子

```python
import ...
data_dir  ="/home/user/Dataset/"
dataset = MyOwnDataset(data_dir)
model = Network(device,param1,param2,...) #自定义的网络
for epoch in range(num_epochs):
    for i in rang(len(dataset)):
        data,label=dataset[i]
        scores=model(data,label)
        loss=criteria(scores,label)
        optimize.zero_grad()#历史梯度清零
        loss.backward()#计算梯度
        optimize.step()#更新权重
        acc = accu(label,target)#计算准确率
    ...
    ...
```

如果需要对数据进行一定的调整，例如压缩，数据增强等操作，就需要Transform 来自

class: torchvision.transformer

tranformer主要代码：

```python
import torchvision.transform as trans

transform = trans.Compose([trans.RandomResized(224),trans.ToTensor()])#融合多种数据处理方式
dataset=MyOwnDataset(data_dir,transform=transform)
...
...
```



**DataLoader讲解**

上面的方式无法对数据进行随机选取，打算顺序，批处理数据等等操作，dataloader就提供这样的方式

dataloader是一个迭代器，code example as following:

```python
import torchvision.transform as trans
from torch.utils.data import DataLoader as DL

transform = trans.Compose([trans.RandomResized(224),trans.ToTensor()])#融合多种数据处理方式
dataset=MyOwnDataset(data_dir,transform=transform)
model=Network(device,param1,...,tranform=transform)
dataloader = DL(dataset,shuffle=True,batch_size=6,num_works=4)
for i,point,label in enumerate(dataloader):
    score=model(point,lable)
    loss = criteria(score,label)
    optimize.zero_grad()
    loss.backward()
    optimiz.step()

```



### 2. Callable object

python 中方法是一种高级类，类一般在使用过程中是先实例化一个对象，然后借用对象调用类中的函数

例如：

```python
class Example:
    def __init__(self,x):
        self.x=x
    def prinf(self):
        for i int range(self.x):
            print("i: ",i)

ex=Example(10)
ex.print()#调用类的函数
```

但是如果重写类的`__call__`函数，就可以将类当作函数使用，例如

```python
class Example(object):
    def __init__(self,name):
        self.name=name
    def __call__(self,x):
        print("x: ",x)
ex = Example("example")
ex(10)#将打印 x: 10
```

### 3.Numpy 

### dimension expansion

主要使用`np.expand_dims( )，包含两个参数，(a,axis),a 是array,axis是要扩充的维度`

example

```python
import numpy as np
a=np.array([[1,1,1],[2,2,2]])#shape 2*3
b=np.expand_dims(a,0)#shape 1*2*3
c=np.expand_dims(a,1)#shape 2*1*3

```

### np array repeats in some dimension

include: `np.tile() np.repeat()`



### 4.KNN

#### knn python

knn的python实现

```python
@staticmethod
def knn(support_pts,query_pts,k):###这种方式20000个点需要30s
    """
        :param support_pts: points you have, B*N1*3
        :param query_pts: points you want to know the neighbour index, B*N2*3
        :param k: Number of neighbours in knn search
        :return: neighbor_idx: neighboring points indexes, B*N2*k
        """
    query_pts_num = query_pts.shape[0]
    idxs = []
    for pt in query_pts:
        all_idx=DataProcessing.single_knn(support_pts,pt)
        idx=[]
        for i in range(k):
            idx.append(all_idx[i])
            idxs.append(idx)
            return np.array(idxs)
        @staticmethod
        def single_knn(support_pts,target):
            dataSetSize = support_pts.shape[0]
            # 将目标数据复制dataSetSize份，然后计算欧式距离
            diffMat = np.tile(target, (dataSetSize, 1)) - support_pts
            sqDiffMat = diffMat ** 2
            distances = sqDiffMat.sum(axis=1)
            # 按照距离进行排序，排序结果为索引
            sortedDistIndicies = distances.argsort()
            # print(type(sortedDistIndicies))
            return sortedDistIndicies
```







