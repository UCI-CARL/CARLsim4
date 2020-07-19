#!/usr/bin/env sh

swig -c++ -python carlsim.i
mkdir -p bin/
mv carlsim.py bin/

# TO BUILD FOR PYTHON2 WITHOUT CUDA SUPPORT
#python2 setup.py build_ext -b ./bin/ -t ./bin/wrap --define __NO_CUDA__ --include $CARLSIM5_INSTALL_DIR/include:/usr/local/cuda/include:/usr/local/cuda/samples/common/inc

# TO BUILD FOR PYTHON3 WITHOUT CUDA SUPPORT
# swig -c++ -python -py3 carlsim.i
python3 setup.py build_ext -b ./bin/ -t ./bin/wrap --define __NO_CUDA__ --include $CARLSIM5_INSTALL_DIR/include:/usr/local/cuda/include:/usr/local/cuda/samples/common/inc
