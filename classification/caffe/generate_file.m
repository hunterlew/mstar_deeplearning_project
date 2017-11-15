clear;
clc;

cd '.\data\mstar';
files = {'train_aug', 'val_aug'};

for k = 1:2
	file_name = files{k};
	fid = fopen([file_name '.txt'], 'w');
	cd(file_name);
	for i = 0:9
	    img_list = dir(num2str(i));
	    for j = 3:length(img_list)
	        fprintf(fid, [num2str(i) '\\' img_list(j).name ' ' num2str(i)]);
	        fprintf(fid, '\n');
	    end
	end
	cd '..'
end
cd '..\..'