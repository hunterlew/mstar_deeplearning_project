import numpy as np
import os
import sys
from matplotlib import pyplot as plt

def show_feature(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    print(data.shape)
    plt.imshow(data[:, :, 0], cmap='gray')
    plt.axis('off')
    plt.show()

# 添加pycaffe路径
root_dir = os.getcwd()
os.chdir('.\\caffe\\build\\Release')
sys.path.append('.\pycaffe')
import caffe

# 初始化网络
net_model = root_dir + '\caffe\examples\mstar\mstar_deploy_96.prototxt'
net_weights = root_dir + '\caffe\examples\mstar\mstar_96_iter_51600.caffemodel'
net = caffe.Net(net_model, net_weights, caffe.TEST)

weight = net.params["conv1"][0].data
print(weight.shape)
show_feature(weight.transpose(0, 2, 3, 1))