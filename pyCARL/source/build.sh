#!/usr/bin/env sh

swig -c++ -python carlsim.i
mkdir -p bin/
mv carlsim.py bin/

# TO BUILD FOR PYTHON2
#python2 setup.py build_ext -b ./bin/ -t ./bin/wrap --define __NO_CUDA__ --include /home/adarsha/CARL/include:/usr/local/cuda/include:/usr/local/cuda/samples/common/inc

# TO BUILD FOR PYTHON3
# swig -c++ -python -py3 carlsim.i
python3 setup.py build_ext -b ./bin/ -t ./bin/wrap --define __NO_CUDA__ --include /home/adarsha/CARL/include:/usr/local/cuda/include:/usr/local/cuda/samples/common/inc

# TO BUILD FOR CUDA SUPPORT
# TODO: Figure out what needs including to fix errors
#python3 setup.py build_ext -b ./bin/ -t ./bin/wrap --define __CUDA9__ --include /home/adarsha/CARL/include:/usr/local/cuda/include:/usr/local/cuda/samples/common/inc
