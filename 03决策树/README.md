# 决策树

## 计算给定数据集的香农熵
其中的信息增益值：当数据集某一特征的香农熵越大代表该特征的影响越大
$$
H=-\sum_{i=1}^{n} p(x_i) log_{2}{p(x_i)}
$$


```python
from math import log
def calcShannonEnt(dataSet):
    # 获取数据集的长度
    numEntries = len(dataSet)
    
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1] # 最后一项为分类结果
        # 如果当前遍历的这一行数据所在分类不在标签的集合中
        if currentLabel not in labelCounts.keys():
            # 在字典中增加对应项
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    print(numEntries)
    print(labelCounts)
    # 最终得到的是一个字典，字典中的值为    
    # 计算
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries # 概率值
        shannonEnt -= prob * log(prob,2)
    return shannonEnt
```


```python
def createTestDataSet():
    dataSet = [
        [1,1,'y'],
        [1,1,'y'],
        [1,0,'n'],
        [0,1,'n'],
        [0,1,'n']
    ]
    labels = ['no surfacing', 'flippers']
    return dataSet,labels
```


```python
myDat,labels = createTestDataSet()
calcShannonEnt(myDat)
```

    5
    {'y': 2, 'n': 3}
    




    0.9709505944546686




```python
myDat[2][-1] = 'm'
myDat
```




    [[1, 1, 'y'], [1, 1, 'y'], [1, 0, 'm'], [0, 1, 'n'], [0, 1, 'n']]




```python
calcShannonEnt(myDat)
```




    1.3931568569324173



## 划分数据集
按照最大的信息增益方法划分数据集


```python
# 需要划分的数据集、指定需要划分的哪一列、那一列对应的哪一个值作为划分依据
def splitDataSet(dataSet,axis,value):
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatvec = featVec[:axis]
            reducedFeatvec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatvec)
    return retDataSet
```


```python
myDat[2][-1] = 'n'
splitDataSet(myDat,0,1)
```




    [[1, 'y'], [1, 'y'], [0, 'n']]




```python
splitDataSet(myDat,0,0)
```




    [[1, 'n'], [1, 'n']]



## 选择最好的数据集划分方式
遍历整个数据集，并循环计算香农熵和分割数据集的函数，找到最好的特征划分方式


```python
def chooseBestFeatureToSplit(dataSet):
    # 特征数
    numFeatures = len(dataSet[0]) -1
    # 计算初始香农熵
    baseEntropy = calcShannonEnt(dataSet)
    # 初始化参数：最好的信息熵，最好的特征
    bestInfoGain = 0.0
    bestFeature = -1
    
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        # 获取不重复的特征值列表
        uniqueValue = set(featList)
        newEntropy = 0.0
        for value in uniqueValue:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature
```


```python
chooseBestFeatureToSplit(myDat)
```




    1



## 构建决策树


```python
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
            classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(),
    key = operator.itemgetter(i),reverse=True)
    return sortedClassCount[0][0]
```


```python
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree
```


```python
myTree = createTree(myDat,labels)
myTree
```




    {'flippers': {0: 'n', 1: {'no surfacing': {0: 'n', 1: 'y'}}}}


