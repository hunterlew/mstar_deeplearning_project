1. 改名my.log
2. python parse_log.py my.log ./ 
	用2.7版本，生成train和test解析
3. 去掉解析文件前两行
4. python plot_training_log.py 2 save.png my.log
	0 lr-it
	1 lr-sec
	2 val_acc-it
	3 val_acc-sec
	4 train_loss-it
	5 train_loss-sec
	
