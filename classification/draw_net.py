#!/usr/bin/env python
"""
Draw a graph of the net architecture.
"""
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from google.protobuf import text_format

# 添加pycaffe路径
import os
import sys

root_dir = os.getcwd()
os.chdir('.\\caffe\\build\\Release')
sys.path.append('.\pycaffe')
import caffe
import caffe.draw
from caffe.proto import caffe_pb2

os.chdir(root_dir)
net = caffe_pb2.NetParameter()
input_net_proto_file = root_dir + '\caffe\examples\mstar\mstar_deploy_96.prototxt'
output_image_file = 'net.pdf'
text_format.Merge(open(input_net_proto_file).read(), net)
print('Drawing net to %s' % output_image_file)
rankdir = 'TB'
# 'One of TB (top-bottom, i.e., vertical), '
#                           'RL (right-left, i.e., horizontal), or another '
#                           'valid dot option; see '
#                           'http://www.graphviz.org/doc/info/'
#                           'attrs.html#k:rankdir'
phase=caffe.TEST;
caffe.draw.draw_net_to_file(net, output_image_file, rankdir, phase)
# 如果报错，GraphViz's executable "dot" not found，请先安装并加入系统环境变量，并重启ide
# www.graphviz.org