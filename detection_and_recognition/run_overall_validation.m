close all;
clc;
% 不能全部clear，需要保留本函数句柄
clear is_valid_handle; 

addpath(genpath('.\core'));
% 待测试图像
load '.\core\imdb\cache\imdb_voc_2007_test.mat';
img_list = imdb.image_ids;

cd .\core
% 添加路径 '.\external\caffe\matlab\caffe_faster_rcnn';
active_caffe_mex(1, 'caffe_faster_rcnn');
% 设置gpu
caffe.set_mode_gpu();
% 输出日志
% caffe.init_log(fullfile(pwd, 'caffe_log'));

% 加载配置文件
model_dir = fullfile(pwd, 'output', 'faster_rcnn_final', 'faster_rcnn_VOC2007_ZF');
load(fullfile(model_dir, 'model.mat'));

% 加载检测网络模型和参数
rpn_net = caffe.Net(fullfile(model_dir, 'proposal_test.prototxt'), 'test');
rpn_net.copy_from(fullfile(model_dir, 'proposal_final'));
% 加载分类网络模型和参数
fast_rcnn_net = caffe.Net(fullfile(model_dir, 'detection_test.prototxt'), 'test');
fast_rcnn_net.copy_from(fullfile(model_dir, 'detection_final'));
% gpu数组形式的均值文件
model.conf_proposal.image_means = gpuArray(model.conf_proposal.image_means);
model.conf_detection.image_means = gpuArray(model.conf_detection.image_means);
% 非极大值抑制参数
nms_thres = 0.6;    % 初筛大一点
nms_num  = 30;
nums_thres_again = 0.3;
% ----该参数下指标----
% Mean time used: 0.028s
% Real target number: 375
% Missing alarm: 2.1%
% False alarm: 0.3%
% Recognition rate: 90.7%

% 预加载热身，有利于更好地计算时间
for j = 1:2 
    im = gpuArray(uint8(ones(375, 500, 3)*128));
    % 检测
    [boxes, scores] = proposal_im_detect(model.conf_proposal, rpn_net, im);
    % 筛选
    aboxes = boxes_filter([boxes, scores], nms_thres, nms_num);
    % 分类
    [boxes, scores] = fast_rcnn_conv_feat_detect(model.conf_detection, fast_rcnn_net, im, ...
        rpn_net.blobs(model.last_shared_output_blob_name), ...
        aboxes(:, 1:4), nms_num);     
end

time_sum = 0;
for j = 1:length(img_list)
    % 计时
    th = tic();
    im = gpuArray(imread(['.\datasets\VOCdevkit2007\VOC2007\JPEGImages\' char(img_list(j)) '.jpg']));
    % 检测
    [boxes, scores] = proposal_im_detect(model.conf_proposal, rpn_net, im);
    % 初次筛选
    aboxes = boxes_filter([boxes, scores], nms_thres, nms_num);
    % 分类
    [boxes, scores] = fast_rcnn_conv_feat_detect(model.conf_detection, fast_rcnn_net, im, ...
        rpn_net.blobs(model.last_shared_output_blob_name), ...
        aboxes(:, 1:4), nms_num);    
    boxes_cell = cell(length(model.classes), 1);
    for i = 1:length(boxes_cell)
        boxes_cell{i} = [boxes(:, (1+(i-1)*4):(i*4)), scores(:, i)];
        boxes_cell{i} = boxes_cell{i}(nms(boxes_cell{i}, nums_thres_again), :);
        % 第二次筛选，只保留分类得分大于60的
        I = boxes_cell{i}(:, 5) >= 0.5;
        boxes_cell{i} = boxes_cell{i}(I, :);
    end
    this_time = toc(th);
    fprintf('%s : %.3fs \n', [char(img_list(i)) '.jpeg'], this_time);
    time_sum = time_sum + this_time;
    % 保存，用于计算指标
    allboxes{j} = [];
    for i = 1:length(boxes_cell)
        if ~isempty(boxes_cell{i})
            allboxes{j} = [allboxes{j}; [boxes_cell{i}, ones(size(boxes_cell{i}, 1), 1)*i]];
        end
    end
end

caffe.reset_all(); 
clear mex;
cd ..
rmpath(genpath('.\core'));

% 计算指标
compute_evaluations(img_list, allboxes, time_sum);

function aboxes = boxes_filter(aboxes, nms_thres, nms_num)
    % 非极大值抑制
    aboxes = aboxes(nms(aboxes, nms_thres, 1), :);    
    % 根据得分排序，最高的N个
    aboxes = aboxes(1:min(length(aboxes), nms_num), :);
end
