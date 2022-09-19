import os
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
# from metrics import masked_mape_np
from scipy.sparse.linalg import eigs
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
root_data_path = '/home/zw100/广义时空回归图卷积神经网络/GSTRGCT/data'
fig_path = '/home/zw100/广义时空回归图卷积神经网络/GSTRGCT/fig'
root_data_path = '/home/zw100/广义时空回归图卷积神经网络/GSTRGCT/data'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
def transform_numpy_to_tensor(data, device=device):
   data = torch.from_numpy(data).type(torch.FloatTensor).to(device)
   return data





class DataProcess:
    def __init__(self, root_data_path , data_file_name, type="pems04"):
        """"
        param: data_file_name-->pems04.npz/pems08.npz绝对路径,shape(16992, 307, 3),(17856, 170, 3)
        """
        self.root_data_path = root_data_path
        self.data_file_name = data_file_name
        self.type = type

    def max_min_normalization(self, x, _max, _min):
        x = 1. * (x - _min) / (_max - _min)
        x = x * 2. - 1.
        return x

    def re_max_min_normalization(self, x, _max, _min):
        x = 1. * x * (_max - _min) + _min
        return x

    def load_data(self): #暂时不变化
        scale = MinMaxScaler()
        scale_data = []
        _min = []
        _max = []
        data = np.load(self.data_file_name)
        data = data['data']
        # 将数据进行标准化
        for i in range(data.shape[0]):
            scale_data.append(scale.fit_transform(data[i, :, :].reshape(-1, 3)))
            _min.append(data[i, :, :].reshape(-1, 3).min(axis=0).reshape(-1,3))
            _max.append(data[i, :, :].reshape(-1, 3).max(axis=0).reshape(-1,3))

        return data, np.array(scale_data), np.array(_min), np.array(_max)

    def generate_dataset(self, scale_data, num_time_steps_input, num_time_steps_output, role="traffic flow prediction"):
        """
        :param scale_data: # 标准化的数据[0,1]
        :param num_time_steps_input:输入时间步数
        :param num_time_steps_output: 预测时间步数
        :param role: “模型用作交通流预测”，否则应该重新生成样本集
        :return: 生成样本集X,Y-->tensor张量，
        """
        X = scale_data #维度为（总的时间序列长度, 节点数, 特征数） #特征数也叫通道数，在统计学中3个通道数实际是独立的特征，因此标准化没有问题
        features, target = [],[]
        if role == "traffic flow prediction":
            indices = [(i, i + (num_time_steps_input + num_time_steps_output)) for i in range(X.shape[0] - (num_time_steps_input + num_time_steps_output) + 1)] #产生样本集的其实索引与终止y的索引
            # 注意i:i+ num_timesteps_input为x序列索引，i+ num_timesteps_input:(i + num_time_steps_input) + num_time_steps_output为预测y的索引

            # 取出序列索引(0,0+num_timesteps_input + num_timesteps_output),(1,1+num_timesteps_input + num_timesteps_output)
            # (34272-num_timesteps_input + num_timesteps_output),34272)
            # 长度为34272-num_timesteps_input + num_timesteps_output+1

            # Save samples
            for i, j in indices:
                features.append(
                    X[i: i + num_time_steps_input, :, :])  # 取出第i个步长为num_timesteps_input的序列
                target.append(X[i + num_time_steps_input: j, :, 0].reshape(num_time_steps_output,-1,1))  # 为什么是0，目标是流量,对于多维时间序列预测，预测的特征，在过去也是输入的特征,reshape 保证X,y维度个数一致
                ##取出第i个预测步长为num_timesteps_output的序列，维度为（207，num_timesteps_input，2）
        else:
            print("该模型不是用于交通流预测，请重新生成样本集")

        return np.array(features), np.array(target) # X:维度为(样本个数(序列个数）,输入时间步数，节点数，特征数）Y:维度为(样本个数(序列个数）,输出时间步数，节点数，特征数=1）
    def get_train_valid_data(self, all_features, all_target, _min, _max, split_size_1=0.6, split_size_2=0.2):
        """
        :param all_features: X样本集,维度为(样本个数(序列个数）,输入时间步数，节点数，特征数
        :param all_target: Y样本集,维度为(样本个数(序列个数）,输出时间步数，节点数，特征数=1）
        :param split_size_1: 划分训练集的样本比例
        :param split_size_2: 划分测试集的样本比例
        :return: all_data-->json
        """
        split_1 = int(len(all_features)*split_size_1)
        split_2 = int(len(all_features)*(split_size_1 +split_size_2))

        train_x = all_features[:split_1, :, :, :]
        val_x = all_features[split_1:split_2, :, :, :]
        test_x = all_features[split_2:, :, :, :]

        train_y = all_target[:split_1, :, :, :]
        val_y = all_target[split_1:split_2, :, :, :]
        test_y = all_target[split_2:, :, :, :]
        all_data = {
            'train': {
                'x': train_x,
                'y': train_y
            },
            'val': {
                'x': val_x,
                'y': val_y
            },
            'test': {
                'x': test_x,
                'y': test_y
            },
            'stats': {
                'min': _min,
                'max': _max
            }
        }
        return all_data

    def get_data_loader(self, all_data, save=True):
        train_x = all_data['train']['x']
        train_y = all_data['train']['y']

        val_x = all_data['val']['x']
        val_y = all_data['val']['y']

        test_x = all_data['test']['x']
        test_y = all_data['test']['y']

        _min = all_data['stats']['min']
        _max = all_data['stats']['max']

        # ------train_loader------
        train_x_tensor = transform_numpy_to_tensor(train_x)
        train_y_tensor = transform_numpy_to_tensor(train_y)
        train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_y_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # ------val_loader------vals[0].real
        val_x_tensor = transform_numpy_to_tensor(val_x)
        val_y_tensor = transform_numpy_to_tensor(val_y)
        val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_y_tensor)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # ------test_loader------
        test_x_tensor = transform_numpy_to_tensor(test_x)
        test_y_tensor = transform_numpy_to_tensor(test_y)
        test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_y_tensor)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # ------print size------
        print('train size', train_x_tensor.size(), train_y_tensor.size())
        print('val size', val_x_tensor.size(), val_y_tensor.size())
        print('test size', test_x_tensor.size(), test_y_tensor.size())
        all_data_loader: dict[str, Union[dict[str, Any], dict[str, Any], dict[str, Union[int, Any]], dict[str, Any]]] = {
            'train': {
                'x_tensor': train_x_tensor,
                'y_tensor': train_y_tensor,
                'data_loader': train_loader
            },
            'val': {
                'x_tensor': val_x_tensor,
                'y_tensor': val_y_tensor,
                'data_loader': val_loader
            },
            'test': {
                'x_tensor': test_x_tensor,
                'y_tensor': test_y_tensor,
                'data_loader': test_loader
            },
            'stats': {
                'min': _min,
                'max': _max,
                'batch_size': batch_size
            }
        }
        if save:
            filename = self.root_data_path + '/%s/train_valid_test_loader_%s.npy' % (self.type, batch_size)
            np.save(filename, all_data_loader)
        return all_data_loader

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


def plot_data(data, time_index=0):
    data = data[time_index,:,:].reshape(-1,3)
    scale = MinMaxScaler()
    data = scale.fit_transform(data)
    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    # ax = Axes3D(fig)
    # ax = plt.subplot(projection='3d')

    ax.scatter(data[:,0],data[:,1],data[:,2])
    plt.savefig(fig_path+'/data_%s.png' % time_index,dpi=300,bbox_inches='tight')

# A_distance平滑得到权重
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

def weight_plot(W):
    ax = sns.heatmap(W)
    ax.set_xlabel('Station ID')
    ax.set_ylabel('Station ID')
    plt.savefig(fig_path +'/W.png', dpi=300, bbox_inches='tight')

def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1] # 确保是方阵

    D = np.diag(np.sum(W, axis=1)) # 计算度

    L = D - W
    # scipy.sparse.linalg.eigs(A,k,M,sigma,which='',..)求稀疏矩阵A的k个特征值和特征向量
    lambda_max = eigs(L, k=1, which='LR')[0].real #最大的实数部分特征值

    return (2 * L) / lambda_max - np.identity(W.shape[0]) # 波浪符号的L：切比雪夫多项式的自变量


def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)

    K: the maximum order of chebyshev polynomials

    Returns
    ----------
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}

    '''

    N = L_tilde.shape[0]

    cheb_polynomials = [np.identity(N), L_tilde.copy()]          # 0阶，1阶

    for i in range(2, K):   #2阶，k-1阶
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials






def compute_val_loss_mstgcn(net, val_loader, criterion, sw, epoch, limit=None):
    '''
    for rnn, compute mean loss on validation set
    :param net: model
    :param val_loader: torch.utils.data.utils.DataLoader
    :param criterion: torch.nn.MSELoss
    :param sw: tensorboardX.SummaryWriter
    :param global_step: int, current global_step
    :param limit: int,
    :return: val_loss
    '''

    net.train(False)  # ensure dropout layers are in evaluation mode

    with torch.no_grad():

        val_loader_length = len(val_loader)  # nb of batch

        tmp = []  # 记录了所有batch的loss

        for batch_index, batch_data in enumerate(val_loader):
            encoder_inputs, labels = batch_data
            outputs = net(encoder_inputs)
            loss = criterion(outputs, labels)  # 计算误差
            tmp.append(loss.item())
            if batch_index % 100 == 0:
                print('validation batch %s / %s, loss: %.2f' % (batch_index + 1, val_loader_length, loss.item()))
            if (limit is not None) and batch_index >= limit:
                break

        validation_loss = sum(tmp) / len(tmp)
        sw.add_scalar('validation_loss', validation_loss, epoch)
    return validation_loss


def evaluate_on_test_mstgcn(net, test_loader, test_target_tensor, sw, epoch, _mean, _std):
    '''
    for rnn, compute MAE, RMSE, MAPE scores of the prediction for every time step on testing set.

    :param net: model
    :param test_loader: torch.utils.data.utils.DataLoader
    :param test_target_tensor: torch.tensor (B, N_nodes, T_output, out_feature)=(B, N_nodes, T_output, 1)
    :param sw:
    :param epoch: int, current epoch
    :param _mean: (1, 1, 3(features), 1)
    :param _std: (1, 1, 3(features), 1)
    '''

    net.train(False)  # ensure dropout layers are in test mode

    with torch.no_grad():

        test_loader_length = len(test_loader)

        test_target_tensor = test_target_tensor.cpu().numpy()

        prediction = []  # 存储所有batch的output

        for batch_index, batch_data in enumerate(test_loader):

            encoder_inputs, labels = batch_data

            outputs = net(encoder_inputs)

            prediction.append(outputs.detach().cpu().numpy())

            if batch_index % 100 == 0:
                print('predicting testing set batch %s / %s' % (batch_index + 1, test_loader_length))

        prediction = np.concatenate(prediction, 0)  # (batch, T', 1)
        prediction_length = prediction.shape[2]

        for i in range(prediction_length):
            assert test_target_tensor.shape[0] == prediction.shape[0]
            print('current epoch: %s, predict %s points' % (epoch, i))
            mae = mean_absolute_error(test_target_tensor[:, :, i], prediction[:, :, i])
            rmse = mean_squared_error(test_target_tensor[:, :, i], prediction[:, :, i]) ** 0.5
            mape = masked_mape_np(test_target_tensor[:, :, i], prediction[:, :, i], 0)
            print('MAE: %.2f' % (mae))
            print('RMSE: %.2f' % (rmse))
            print('MAPE: %.2f' % (mape))
            print()
            if sw:
                sw.add_scalar('MAE_%s_points' % (i), mae, epoch)
                sw.add_scalar('RMSE_%s_points' % (i), rmse, epoch)
                sw.add_scalar('MAPE_%s_points' % (i), mape, epoch)


def predict_and_save_results_mstgcn(net, data_loader, data_target_tensor, global_step, _mean, _std, params_path, type):
    '''

    :param net: nn.Module
    :param data_loader: torch.utils.data.utils.DataLoader
    :param data_target_tensor: tensor
    :param epoch: int
    :param _mean: (1, 1, 3, 1)
    :param _std: (1, 1, 3, 1)
    :param params_path: the path for saving the results
    :return:
    '''
    net.train(False)  # ensure dropout layers are in test mode

    with torch.no_grad():

        data_target_tensor = data_target_tensor.cpu().numpy()

        loader_length = len(data_loader)  # nb of batch

        prediction = []  # 存储所有batch的output

        input = []  # 存储所有batch的input

        for batch_index, batch_data in enumerate(data_loader):

            encoder_inputs, labels = batch_data

            input.append(encoder_inputs[:, :, 0:1].cpu().numpy())  # (batch, T', 1)

            outputs = net(encoder_inputs)

            prediction.append(outputs.detach().cpu().numpy())

            if batch_index % 100 == 0:
                print('predicting data set batch %s / %s' % (batch_index + 1, loader_length))

        input = np.concatenate(input, 0)

        input = re_normalization(input, _mean, _std)

        prediction = np.concatenate(prediction, 0)  # (batch, T', 1)

        print('input:', input.shape)
        print('prediction:', prediction.shape)
        print('data_target_tensor:', data_target_tensor.shape)
        output_filename = os.path.join(params_path, 'output_epoch_%s_%s' % (global_step, type))
        np.savez(output_filename, input=input, prediction=prediction, data_target_tensor=data_target_tensor)

        # 计算误差
        excel_list = []
        prediction_length = prediction.shape[2]

        for i in range(prediction_length):
            assert data_target_tensor.shape[0] == prediction.shape[0]
            print('current epoch: %s, predict %s points' % (global_step, i))
            mae = mean_absolute_error(data_target_tensor[:, :, i], prediction[:, :, i])
            rmse = mean_squared_error(data_target_tensor[:, :, i], prediction[:, :, i]) ** 0.5
            mape = masked_mape_np(data_target_tensor[:, :, i], prediction[:, :, i], 0)
            print('MAE: %.2f' % (mae))
            print('RMSE: %.2f' % (rmse))
            print('MAPE: %.2f' % (mape))
            excel_list.extend([mae, rmse, mape])

        # print overall results
        mae = mean_absolute_error(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1))
        rmse = mean_squared_error(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1)) ** 0.5
        mape = masked_mape_np(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0)
        print('all MAE: %.2f' % (mae))
        print('all RMSE: %.2f' % (rmse))
        print('all MAPE: %.2f' % (mape))
        excel_list.extend([mae, rmse, mape])
        print(excel_list)
if __name__ == "__main__":
    # A, A_distance, dist_mean, dist_std=get_adjacency_matrix(root_data_path + '/pems08/distance.csv', 307)
    # print(A.shape, distanceA.shape)
    dp = DataProcess(root_data_path, data_file_name=root_data_path + '/pems04/pems04.npz', type='pems04')
    data, scale_data, _min, _max = dp.load_data()
    all_features, all_target = dp.generate_dataset(scale_data, num_time_steps_input=12, num_time_steps_output=3,
                                                   role="traffic flow prediction")
    all_data = dp.get_train_valid_data(all_features, all_target, _min, _max, split_size_1=0.6, split_size_2=0.2)
    all_data_loader = dp.get_data_loader(all_data, save=True)

    # W = weight_matrix(A, A_distance, dist_mean, dist_std, scaling=True)
    # weight_plot(W)
    # plot_data(data, time_index=24)
    # print(W)
    print(data.shape)
    # print(scale_data)
    print(scale_data.shape)
    print(_min.shape)
    print(_max.shape)
    print(all_features.shape)
    print(all_target.shape)




