set RESIZE_HEIGHT=96
set RESIZE_WIDTH=96

echo "Creating train_aug lmdb..."

.\build\Release\convert_imageset.exe ^
	--resize_height=%RESIZE_HEIGHT% ^
    --resize_width=%RESIZE_WIDTH% ^
    --shuffle ^
    --gray ^
    .\data\mstar\train_aug\ ^
    .\data\mstar\train_aug.txt ^
    .\examples\mstar\mstar_train_aug_lmdb

echo "Creating val_aug lmdb..."

.\build\Release\convert_imageset.exe ^
	--resize_height=%RESIZE_HEIGHT% ^
    --resize_width=%RESIZE_WIDTH% ^
    --shuffle ^
    --gray ^
    .\data\mstar\val_aug\ ^
    .\data\mstar\val_aug.txt ^
    .\examples\mstar\mstar_val_aug_lmdb

pause