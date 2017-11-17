clear;
cd caffe
addpath(genpath('.\build\Release'));

% 设置gpu或cpu模式
% caffe.set_mode_cpu();
caffe.set_mode_gpu();
caffe.set_device(0);

% 设置solver
% solver = caffe.Solver('.\examples\mstar\mstar_solver.prototxt');
solver = caffe.Solver('.\examples\mstar\mstar_solver_96.prototxt');
% 训练
tic;
solver.solve();
toc;
rmpath(genpath('.\build\Release'));
cd ..