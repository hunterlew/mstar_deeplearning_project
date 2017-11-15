clear;
cd caffe-Release
addpath(genpath('.\build\Release'));

% 设置gpu或cpu模式
% caffe.set_mode_cpu();
caffe.set_mode_gpu();
caffe.set_device(0);

% 设置solver
solver = caffe.Solver('.\examples\mstar\mstar_solver.prototxt');
% 训练
solver.solve();
rmpath(genpath('.\build\Release'));
cd ..