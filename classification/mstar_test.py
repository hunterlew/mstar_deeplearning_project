import numpy as np
import os
import sys
import time
from skimage import io

# 添加pycaffe路径
root_dir = os.getcwd()
os.chdir('.\\caffe\\build\\Release')
sys.path.append('.\pycaffe')
import caffe

# 设置gpu或cpu
# caffe.set_mode_cpu()
caffe.set_mode_gpu()
caffe.set_device(0)

# 读取图片
gray = True
im = io.imread(root_dir + '\\100018_96.jpeg', as_gray=gray)
# 模仿由matlab的亮度操作...坑
im = np.int32(im) * 4
im[im>255] = 255

# 预处理
im = np.float32(im) / 255.0
im -= np.mean(im)
im = im[:, :, np.newaxis]
im = np.transpose(im, (2, 0, 1))
input_data = im

# 初始化网络
net_model = root_dir + '\caffe\examples\mstar\mstar_deploy_96.prototxt'
net_weights = root_dir + '\caffe\examples\mstar\mstar_96_iter_34400.caffemodel'
net = caffe.Net(net_model, net_weights, caffe.TEST)
# 热身
net.blobs['data'].data[...] = np.ones([1, 1, 96, 96])
output = net.forward()

net.blobs['data'].data[...] = input_data
# win下用clock
start = time.clock()
# 前向计算
output = net.forward()['ip3'][0]
print('Time used: ' + str(round((time.clock() - start), 6)) + 's')
print(output)
print(np.max(output), np.argmax(output))