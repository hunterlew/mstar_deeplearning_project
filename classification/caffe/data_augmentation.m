clear;clc;
cd '.\data\mstar';

if ~exist('train_aug', 'dir')
    copyfile('train', 'train_aug');
end
for i = 0:9
    img_list = dir(['.\train_aug\' num2str(i)]);
    for j = 3:length(img_list)
        if exist(['.\train_aug\' num2str(i) '\' img_list(j).name], 'file')
            im = imread(['.\train_aug\' num2str(i) '\' img_list(j).name]);
%             im = uint8(im * 4);
            for k = 0:9
                x = randi(32);
                y = randi(32);
                im_crop = imcrop(im, [x, y, 95, 95]);
                imwrite(im_crop, ['.\train_aug\' num2str(i) '\' img_list(j).name(1:end-5) '_' num2str(k) '.jpeg']);
            end
            % im0 = imcrop(im, [17,17,95,95]);
            % im1 = imcrop(im, [1,1,95,95]);
            % im2 = imcrop(im, [33,33,95,95]);
            % im3 = imcrop(im, [1,33,95,95]);
            % im4 = imcrop(im, [33,1,95,95]);
            % imwrite(im0, ['.\train_aug\' num2str(i) '\' img_list(j).name(1:end-5) '_0.jpeg']);
            % imwrite(im1, ['.\train_aug\' num2str(i) '\' img_list(j).name(1:end-5) '_1.jpeg']);
            % imwrite(im2, ['.\train_aug\' num2str(i) '\' img_list(j).name(1:end-5) '_2.jpeg']);
            % imwrite(im3, ['.\train_aug\' num2str(i) '\' img_list(j).name(1:end-5) '_3.jpeg']);
            % imwrite(im4, ['.\train_aug\' num2str(i) '\' img_list(j).name(1:end-5) '_4.jpeg']);
            % imwrite(flipud(im0), ['.\train_aug\' num2str(i) '\' img_list(j).name(1:end-5) '_5.jpeg']);
            % imwrite(flipud(im1), ['.\train_aug\' num2str(i) '\' img_list(j).name(1:end-5) '_6.jpeg']);
            % imwrite(flipud(im2), ['.\train_aug\' num2str(i) '\' img_list(j).name(1:end-5) '_7.jpeg']);
            % imwrite(flipud(im3), ['.\train_aug\' num2str(i) '\' img_list(j).name(1:end-5) '_8.jpeg']);
            % imwrite(flipud(im4), ['.\train_aug\' num2str(i) '\' img_list(j).name(1:end-5) '_9.jpeg']);
        end
        delete(['.\train_aug\' num2str(i) '\' img_list(j).name]);
    end
end

if ~exist('val_aug', 'dir')
    copyfile('val', 'val_aug');
end
for i = 0:9
    img_list = dir(['.\val_aug\' num2str(i)]);
    for j = 3:length(img_list)
        if exist(['.\val_aug\' num2str(i) '\' img_list(j).name], 'file')
            im = imread(['.\val_aug\' num2str(i) '\' img_list(j).name]);
%             im = uint8(im * 4);
            im = imcrop(im, [17,17,95,95]);
            imwrite(im, ['.\val_aug\' num2str(i) '\' img_list(j).name]);
        end
    end
end

cd '..\..'