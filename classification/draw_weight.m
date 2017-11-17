clear;
clc;
close all;
cd caffe
addpath(genpath('.\build\Release'));

caffe.set_mode_gpu();
% 加载网络文件
net_model = '.\examples\mstar\mstar_deploy_96.prototxt';
% 加载参数文件
net_weights = '.\examples\mstar\mstar_96_iter_51600.caffemodel';
phase = 'test';
% 初始化网络
net = caffe.Net(net_model, net_weights, phase);

param_names={'conv1', 'conv2'};
for i=1:length(param_names)
    visualize_weight(net,param_names{i},1);
end
cd ..

function visualize_weight(net,param_name,space)
    w=net.params(param_name,1).get_data();
    size(w)
    nums=size(w,4);
    channels=size(w,3);
    width=size(w,2);
    count=nums*channels;
    n=ceil(sqrt(count));
    weight_map=zeros(n*(width+space),n*(width+space),'uint8');
    w=w-min(w(:));
    w=w/max(w(:))*255;
    w=uint8(w);
    for i=0:count-1
        c=mod(i,n);
        r=floor(i/n);
        j=mod(i,channels)+1;
        k=floor(i/channels)+1;
        weight_map(r*(width+space)+(1:width),c*(width+space)+(1:width))=w(:,:,j,k);
    end
    figure;
    imshow(weight_map, 'initialMagnification', 'fit');
    title(param_name);
end