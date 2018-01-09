close all;
clc;
% 不能全部clear，需要保留本函数句柄
clear is_valid_handle; 
clear;

% 待测试图像
img_list = dir('.\*.jpg');

% ----------------检测------------------
addpath(genpath('.\core'));
cd .\core
% 添加路径 '.\external\caffe\matlab\caffe_faster_rcnn';
active_caffe_mex(1, 'caffe_faster_rcnn');
% 设置gpu
caffe.set_mode_gpu();
% 输出日志
% caffe.init_log(fullfile(pwd, 'caffe_log'));

% 加载配置文件
model_dir = fullfile(pwd, 'output', 'faster_rcnn_final', 'faster_rcnn_VOC2007_ZF');
load(fullfile(model_dir, 'proposal_alone.mat'));
% 加载检测网络模型和参数
rpn_net = caffe.Net(fullfile(model_dir, 'proposal_test_alone.prototxt'), 'test');
rpn_net.copy_from(fullfile(model_dir, 'proposal_alone'));
% gpu数组形式的均值文件
% model.conf_proposal.image_means = gpuArray(conf_proposal.image_means);

rpn_thres = 0.96;
nms_thres  = 0.6;
nms_thres_again = 0.3;

% 预加载热身，有利于更好地计算时间
for j = 1:2 
    im = gpuArray(uint8(ones(375, 500, 3)*128));
    % 检测
    [boxes, scores] = proposal_im_detect(conf_proposal, rpn_net, im);
    % 筛选
    aboxes = boxes_filter([boxes, scores], rpn_thres, nms_thres);
end

for i = 1:length(img_list)
    th = tic();
    im = gpuArray(imread(['..\' img_list(i).name]));
    % 检测
    [boxes, det_scores] = proposal_im_detect(conf_proposal, rpn_net, im);
    aboxes = boxes_filter([boxes, det_scores], rpn_thres, nms_thres);
    [allboxes{i}, proposals{i}] = rois_modified(gather(im), aboxes);
    time{i} = toc(th);
end

caffe.reset_all();
cd ..
rmpath(genpath('.\core'));

% ----------------分类------------------
cd '..\classification\caffe';
addpath(genpath('.\build\Release'));
caffe.set_mode_gpu();
caffe.set_device(0);
% 加载网络文件
net_model = '.\examples\mstar\mstar_deploy_3.prototxt';
% 加载参数文件
net_weights = '.\examples\mstar\mstar_96_3_iter_51600.caffemodel';
phase = 'test';
% 初始化网络
net = caffe.Net(net_model, net_weights, phase);
% 热身
net.forward({ones(96, 96, 1, 1)});    
net.forward({ones(96, 96, 1, 1)});    
 
for i = 1:length(img_list)
    th = tic();
    proposal = proposals{i};
    scores = [];
    classes = [];
    for j = 1:size(allboxes{i}, 1)
        [scores(j), classes(j)] = classify_on_detection(net, proposal(:, :, j));
    end
    reg_scores{i} = [scores' classes'];
    time{i} = time{i} + toc(th);
end
caffe.reset_all();
rmpath(genpath('.\build\Release'));
cd ..\..\detection_and_recognition

time_sum = 0;
addpath(genpath('.\core'));
for i = 1:length(img_list)
    th = tic();
    aboxes = [allboxes{i} reg_scores{i}];
%     aboxes = [allboxes{i} ones(size(allboxes{i}, 1), 2)];
    aboxes = aboxes(nms(aboxes(:, 1:5), nms_thres_again), :);
    aboxes = aboxes(aboxes(:, 5) >= 0.5, :);
    this_time = time{i} + toc(th);
    fprintf('%s : %.3fs \n', img_list(i).name, this_time);
    time_sum = time_sum + this_time;
    % 保存，用于计算指标
    figure(i);
    m_showboxes(imread(img_list(i).name), aboxes);
end
rmpath(genpath('.\core'));

% 区域修正，需要保证输入是96*96
function [patches, proposals] = rois_modified(im, patches)
    im = rgb2gray(im);
%     im = imresize(im, [594, 492]);

    % 精修边框为96*96正方形
    xcen = (patches(:,1)+patches(:,3))/2;
    ycen = (patches(:,2)+patches(:,4))/2;
    % 刚好为左右边界
    patches(:,1) = max(xcen - 47,1);
    patches(:,3) = min(xcen + 48,size(im,2));
    % 刚好为上下边界
    patches(:,2) = max(ycen - 47,1);
    patches(:,4) = min(ycen + 48,size(im,1));

    for i = 1:size(patches, 1)
        proposals(:,:,i) = imresize(im(round(patches(i,2)) : round(patches(i,4)), ...
                                    round(patches(i,1)) : round(patches(i,3))),[96 96]);    
    end
    patches = patches(:, 1:end-1);
end

function aboxes = boxes_filter(aboxes, rpn_thres, nms_thres)
    % 由于分类网络没有负样本，按照共享参数模型的过滤方法会有很多虚惊
    % 因此对rpn的要求比较高，需要挑选得分较高的分数
    % 根据得分排序，最高的N个
    aboxes = aboxes(aboxes(:, 5) >= rpn_thres, :);   
    % 非极大值抑制
    aboxes = aboxes(nms(aboxes, nms_thres, 1), :);    
end

function [score, class] = classify_on_detection(net, im)
    % gpu热身
    im = single(im) / 255.0;
    im = im - mean(im(:));
    im = permute(im, [2, 1, 3]);
    im = reshape(im, [96, 96, 1, 1]);
    scores = net.forward({im});
    scores = scores{1};
    [score, class] = max(scores);
end