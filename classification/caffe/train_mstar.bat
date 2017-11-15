SET GLOG_logtostderr=0
SET GLOG_log_dir=.\examples\mstar\Log\
.\build\Release\caffe.exe train --solver=.\examples\mstar\mstar_solver.prototxt
pause