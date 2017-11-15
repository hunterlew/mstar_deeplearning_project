clc;
clear mex;
clear is_valid_handle; 

% 如果重新训练，需要删除cache和output
delete('.\imdb\cache\*');
out_dir = dir('.\output');
for i = 3:length(out_dir)
    rmdir(fullfile('output', out_dir(i).name), 's');
end

% 添加路径，选择gpu训练
active_caffe_mex(1, 'caffe_faster_rcnn');
caffe.set_mode_gpu();

% 加载训练文件配置
model = Model.ZF_for_Faster_RCNN_VOC2007;
cache_base_proposal = 'faster_rcnn_VOC2007_ZF';
cache_base_fast_rcnn = '';
% 数据集加载，训练集镜像扩充
dataset = [];
dataset = Dataset.voc2007_trainval(dataset, 'train', true);
dataset = Dataset.voc2007_test(dataset, 'test', false);
% 加载均值文件
conf_proposal = proposal_config('image_means', model.mean_image, 'feat_stride', model.feat_stride);
conf_fast_rcnn = fast_rcnn_config('image_means', model.mean_image);
% 设置缓存文件夹
model = Faster_RCNN_Train.set_cache_folder(cache_base_proposal, cache_base_fast_rcnn, model);
% 通过比例映射和anchor机制生成候选框大小 
[conf_proposal.anchors, conf_proposal.output_width_map, conf_proposal.output_height_map] ...
                            = proposal_prepare_anchors(conf_proposal, model.stage1_rpn.cache_name, model.stage1_rpn.test_net_def_file);

% 开始训练                        
fprintf('\n***************\nstage one proposal \n***************\n');
model.stage1_rpn = Faster_RCNN_Train.do_proposal_train(conf_proposal, dataset, model.stage1_rpn, true);
dataset.roidb_train = cellfun(@(x, y) Faster_RCNN_Train.do_proposal_test(conf_proposal, model.stage1_rpn, x, y), dataset.imdb_train, dataset.roidb_train, 'UniformOutput', false);
dataset.roidb_test = Faster_RCNN_Train.do_proposal_test(conf_proposal, model.stage1_rpn, dataset.imdb_test, dataset.roidb_test);
% 如果是分立模型，则训练至此结束；
% 联合模型继续训练

fprintf('\n***************\nstage one fast rcnn\n***************\n');
model.stage1_fast_rcnn = Faster_RCNN_Train.do_fast_rcnn_train(conf_fast_rcnn, dataset, model.stage1_fast_rcnn, true);
opts.mAP = Faster_RCNN_Train.do_fast_rcnn_test(conf_fast_rcnn, model.stage1_fast_rcnn, dataset.imdb_test, dataset.roidb_test);

fprintf('\n***************\nstage two proposal\n***************\n');
% 固定第一阶段共享部分的参数，该部分学习率为0
model.stage2_rpn.init_net_file = model.stage1_fast_rcnn.output_model_file;
model.stage2_rpn = Faster_RCNN_Train.do_proposal_train(conf_proposal, dataset, model.stage2_rpn, true);
dataset.roidb_train       	= cellfun(@(x, y) Faster_RCNN_Train.do_proposal_test(conf_proposal, model.stage2_rpn, x, y), dataset.imdb_train, dataset.roidb_train, 'UniformOutput', false);
dataset.roidb_test       	= Faster_RCNN_Train.do_proposal_test(conf_proposal, model.stage2_rpn, dataset.imdb_test, dataset.roidb_test);

fprintf('\n***************\nstage two fast rcnn\n***************\n');
% 固定第一阶段共享部分的参数，该部分学习率为0
model.stage2_fast_rcnn.init_net_file = model.stage1_fast_rcnn.output_model_file;
model.stage2_fast_rcnn      = Faster_RCNN_Train.do_fast_rcnn_train(conf_fast_rcnn, dataset, model.stage2_fast_rcnn, true);

% 测试
fprintf('\n***************\nfinal test\n***************\n');    
model.stage2_rpn.nms        = model.final_test.nms;
dataset.roidb_test       	= Faster_RCNN_Train.do_proposal_test(conf_proposal, model.stage2_rpn, dataset.imdb_test, dataset.roidb_test);
opts.final_mAP              = Faster_RCNN_Train.do_fast_rcnn_test(conf_fast_rcnn, model.stage2_fast_rcnn, dataset.imdb_test, dataset.roidb_test);

% 保存模型
Faster_RCNN_Train.gather_rpn_fast_rcnn_models(conf_proposal, conf_fast_rcnn, model, dataset);

% anchor机制及box映射
% 由于毕设样本制作和标注比较保守，只需要一种anchor，可以尝试多尺度检测识别
% scales为8，ratio为1，即设置边长为128的框作为计算iou的box
function [anchors, output_width_map, output_height_map] = proposal_prepare_anchors(conf, cache_name, test_net_def_file)
    [output_width_map, output_height_map] ...                           
                                = proposal_calc_output_size(conf, test_net_def_file);
    anchors                = proposal_generate_anchors(cache_name, ...
                                    'scales',  8,...
                                    'ratios',  [1]);
end