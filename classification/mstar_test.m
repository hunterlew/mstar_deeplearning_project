clear;
% 输入，记得转为行优先
im = imread('140003.jpeg');
im = single(im) / 255.0;
im = im - mean(im(:));
im = permute(im, [2, 1, 3]);
im = reshape(im, [96, 96, 1, 1]);
% for i = 1:2000
%     im_2000(:, :, :, i) = im;
% end
% input_data = {im_2000};
input_data = {im};

cd caffe
addpath(genpath('.\build\Release'));

% 设置gpu或cpu模式
% caffe.set_mode_cpu();
caffe.set_mode_gpu();
caffe.set_device(0);

% 加载网络文件
net_model = '.\examples\mstar\mstar_deploy_2.prototxt';
% 加载参数文件
net_weights = '.\examples\mstar\mstar_96_2_iter_51600.caffemodel';
phase = 'test';
% 初始化网络
net = caffe.Net(net_model, net_weights, phase);
% gpu热身
net.forward({ones(96, 96, 1, 1)});

tic;
% 前向计算
scores = net.forward(input_data);
toc;
scores = scores{1};
[score, label] = max(scores);
score
label

% 清理内存
caffe.reset_all();
rmpath(genpath('.\build\Release'));
cd ..