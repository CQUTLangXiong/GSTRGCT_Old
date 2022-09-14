# 广义时空回归图神经网络（GSTRGCT)

该模型主要是作为地理学、气象学、经济学等传统学科的时空建模研究向人工智能方向过渡的学习范式。

## 1. 数据处理

### 1.1 distance的处理

```python
def get_adjacency_matrix(adj_filename, num_of_vertices):
    """_summary_
    Args:
        adj_filename (str): like "F:/广义时空回归图卷积神经网络/GSTRGCT/data/PEMS04/distance.csv"
        num_of_vertices (int): thr number of vertices
    return: adjacency matrix(A) and distance matrix(distanceA)
    """
    # 生成空的邻接矩阵以及距离矩阵
    A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32) 
    A_distance = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)
  
    # 导入csv数据
    distance = pd.read_csv(adj_filename) # columns are "from", "to", and "cost"
    for i in range(len(distance)):
        from_index = distance['from'][i]
        to_index = distance['to'][i]
        cost = distance['cost'][i]
        A[from_index, to_index] = 1
        A_distance[from_index, to_index] = cost
    dist_mean = distance['cost'].mean()
    dist_std = distance['cost'].std()
    return A, A_distance, dist_mean, dist_std
```

生成邻接矩阵A，以及距离矩阵A_distance,对应距离的均值和标准差。将A_distance进行拓扑平滑处理

![1663077769209](image/GSTRGCT/1663077769209.png)


```python
def weight_matrix(A, A_distance, dist_mean, dist_std, scaling=True):
    """
    :param A_distance: 空间距离矩阵
    :param scaling: 决定是否采用此平滑后的权重矩阵，否则使用0/1的A
    :return:空间权重
    """
    if scaling:
        W = A_distance
        # n = A_distance.shape[0]
        # W = W / dist_max #缩放到0到1之间
        # W = W
        # W2, W_mask = W * W, np.ones([n, n]) - np.identity(n)# 对角为0
        # W2 = W * W # 距离的平方
        W = np.exp(-(W-dist_mean)* (W-dist_mean)/ (dist_std*dist_std)) * A # 0非连通的地方权重为0，连通的地方为距离越大权重越小
        # refer to Eq.10
    else:
        W = A
    return W
```

平滑后的空间权重考虑了拓扑连通性以及欧式空间距离（K-means语义距离暂不考虑，聚集效果不是很明显）

![1663078013056](image/GSTRGCT/1663078013056.png)

### 1.2 原始数据的读取

读取原始的数据pems04.npz,pems08.npz

```python
def load_data(data_file_name):
    data = np.load(data_file_name)
    return data['data']
```

pems04维度为(16992, 307, 3)$\rightarrow$（时序长度，节点个数，特征数）

pems08维度为(17856, 170, 3)

### 原始数据集的划分


```python
def generate_dataset(scale_data, num_timesteps_input, num_timesteps_output):
    """
    Takes node features for the graph and divides them into multiple samples
    along the time-axis by sliding a window of size (num_timesteps_input+
    num_timesteps_output) across it in steps of 1.
    :param X: Node features of shape (num_vertices, num_features,
    num_timesteps)
    :return:
        - Node features divided into multiple samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_input).
        - Node targets for the samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_output).
    """
    # Generate the beginning index and the ending index of a sample, which
    # contains (num_points_for_training + num_points_for_predicting) points
    #取出X和y
    X = scale_data
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[0] - (
                num_timesteps_input + num_timesteps_output) + 1)] 
    #取出序列索引(0,0+num_timesteps_input + num_timesteps_output),(1,1+num_timesteps_input + num_timesteps_output)
    #(34272-num_timesteps_input + num_timesteps_output),34272)
    #长度为34272-num_timesteps_input + num_timesteps_output+1

    # Save samples
    features, target = [], []
    for i, j in indices:
        features.append(
            X[i: i + num_timesteps_input, :, :])#取出第i个步长为num_timesteps_input的序列，维度为（207，num_timesteps_input，2）
        target.append(X[i + num_timesteps_input: j, :, 0].reshape(num_timesteps_output,-1,1))#为什么是0，目标是流量
        ##取出第i个预测步长为num_timesteps_output的序列，维度为（207，num_timesteps_input，2）

    return torch.from_numpy(np.array(features)),            torch.from_numpy(np.array(target))
```


## 2. 模型框架
