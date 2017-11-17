import os
import sys
import time

root_dir = os.getcwd()
os.chdir('.\\caffe\\build\\Release')
sys.path.append('.\pycaffe')
import caffe

caffe.set_mode_gpu();
caffe.set_device(0);

os.chdir(root_dir + '.\caffe')
# solver = caffe.SGDSolver('.\examples\mstar\mstar_solver.prototxt');
solver = caffe.SGDSolver('.\examples\mstar\mstar_solver_96.prototxt');
start = time.clock()
solver.solve();
print('Time used: ' + str(round((time.clock() - start), 6)) + 's')