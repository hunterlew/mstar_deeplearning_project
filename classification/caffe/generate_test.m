clear;clc;
cd '.\data\mstar';

if ~exist('test', 'dir')
    copyfile('val', 'test');
end
for i = 0:9
    img_list = dir(['.\test\' num2str(i)]);
    for j = 3:length(img_list)
        if exist(['.\test\' num2str(i) '\' img_list(j).name], 'file')
            im = imread(['.\test\' num2str(i) '\' img_list(j).name]);
%             im = uint8(im * 4);
            x = randi(32);
            y = randi(32);
            im = imcrop(im, [x,y,95,95]);
            imwrite(im, ['.\test\' num2str(i) '\' img_list(j).name]);
        end
    end
end

cd '..\..'