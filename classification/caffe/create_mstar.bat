set RESIZE_HEIGHT=128
set RESIZE_WIDTH=128

echo "Creating train lmdb..."

.\build\Release\convert_imageset.exe ^
	--resize_height=%RESIZE_HEIGHT% ^
    --resize_width=%RESIZE_WIDTH% ^
    --shuffle ^
    --gray ^
    .\data\mstar\train\ ^
    .\data\mstar\train.txt ^
    .\examples\mstar\mstar_train_lmdb

echo "Creating val lmdb..."

.\build\Release\convert_imageset.exe ^
	--resize_height=%RESIZE_HEIGHT% ^
    --resize_width=%RESIZE_WIDTH% ^
    --shuffle ^
    --gray ^
    .\data\mstar\val\ ^
    .\data\mstar\val.txt ^
    .\examples\mstar\mstar_val_lmdb

pause