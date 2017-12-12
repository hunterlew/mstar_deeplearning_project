clear;
clc;
close all;
im = imread('100030.jpeg');
im = imcrop(im, [17,17,95,95]);

cd caffe
addpath(genpath('.\build\Release'));

caffe.set_mode_gpu();
% 加载网络文件
net_model = '.\examples\mstar\mstar_deploy_2.prototxt';
% 加载参数文件
net_weights = '.\examples\mstar\mstar_96_2_iter_51600.caffemodel';
phase = 'test';
% 初始化网络
net = caffe.Net(net_model, net_weights, phase);

im = single(im) / 255.0;
im = im - mean(im(:));
% im = permute(im, [2, 1, 3]);
im = reshape(im, [96, 96, 1, 1]);
scores = net.forward({im});   

param_names={'conv1', 'conv2', 'conv3', 'conv4'};
blobs_names={'conv1', 'conv2', 'conv3', 'conv4'};

for m=1:length(param_names)
    w = net.params(param_names{m},1).get_data();
    w_vec = [];
    w_sum = [];
    w_sorted = [];
    for i = 1:size(w, 4)
        for j = 1:size(w, 3)
            w_vec{size(w, 3)*(i-1)+ j} = w(:,:,j,i);
            tmp = w(:,:,j,i); tmp = tmp(:);
            w_sum(size(w, 3)*(i-1)+ j) = sum(tmp.^2);
        end
    end
    [~, ind] = sort(w_sum, 'descend');
    ind_rand = randperm(size(w, 4));
    ind = ind(ind_rand);
    for i = 1:size(w, 4)
        w_sorted(:,:,i,1) = w_vec{ind(i)};
    end
    weight_map = visualize_weight(net,w_sorted,1);
    imwrite(imresize(weight_map,[255 255]),['w2_' num2str(m) '.jpg']);
end

for m=1:length(blobs_names)
    w = net.blobs(blobs_names{m}).get_data();
    w_vec = [];
    w_sum = [];
    w_sorted = [];
    for i = 1:size(w, 4)
        for j = 1:size(w, 3)
            w_vec{size(w, 3)*(i-1)+ j} = w(:,:,j,i);
            tmp = w(:,:,j,i); tmp = tmp(:);
            w_sum(size(w, 3)*(i-1)+ j) = sum(tmp.^2);
        end
    end
    [~, ind] = sort(w_sum, 'descend');
    ind_rand = randperm(size(w, 3));
    ind = ind(ind_rand);
    for i = 1:size(w, 3)
        w_sorted(:,:,i,1) = w_vec{ind(i)};
    end
    weight_map = visualize_weight(net,w_sorted,1);
    imwrite(imresize(weight_map,[255 255]),['f2_' num2str(m) '.jpg']);
end   

cd ..

function weight_map = visualize_weight(net,w,space)
    size(w)
    nums=size(w,4);
    channels = 1;
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
%     title(param_name);
end