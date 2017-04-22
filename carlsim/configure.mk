##----------------------------------------------------------------------------##
##
##   CARLsim4 Configuration
##   ----------------------
##
##   Authors:   Michael Beyeler <mbeyeler@uci.edu>
##              Kristofor Carlson <kdcarlso@uci.edu>
##
##   Institute: Cognitive Anteater Robotics Lab (CARL)
##              Department of Cognitive Sciences
##              University of California, Irvine
##              Irvine, CA, 92697-5100, USA
##
##   Version:   03/07/2017
##
##----------------------------------------------------------------------------##

#------------------------------------------------------------------------------
# Common paths
#------------------------------------------------------------------------------

# path to CUDA installation
CUDA_PATH        ?= /usr/local/cuda

# enable gcov
CARLSIM4_COVERAGE ?= 0

#------------------------------------------------------------------------------
# CARLsim/ECJ Parameter Tuning Interface Options
#------------------------------------------------------------------------------
# absolute path and name of evolutionary computation in java installation jar
# file for ECJ-PTI CARLsim support.
ECJ_JAR ?= /opt/ecj/jar/ecj.23.jar
ECJ_PTI_DIR ?= /opt/CARL/carlsim_ecj_pti


#------------------------------------------------------------------------------
# Common binaries 
#------------------------------------------------------------------------------
CXX   ?= g++
CLANG ?= /usr/bin/clang
MV    ?= mv -f
RM    ?= rm -f
RMR   ?= rm -rf


#------------------------------------------------------------------------------
# Find OS 
#------------------------------------------------------------------------------
# OS Name (Linux or Darwin)
OSUPPER = $(shell uname -s 2>/dev/null | tr "[:lower:]" "[:upper:]")
OSLOWER = $(shell uname -s 2>/dev/null | tr "[:upper:]" "[:lower:]")

# Flags to detect 32-bit or 64-bit OS platform
OS_SIZE = $(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/" -e "s/armv7l/32/")
OS_ARCH = $(shell uname -m | sed -e "s/i386/i686/")

# search at Darwin (unix based info)
DARWIN = $(strip $(findstring DARWIN, $(OSUPPER)))
ifneq ($(DARWIN),)
   SNOWLEOPARD = $(strip $(findstring 10.6, $(shell egrep "<string>10\.6" /System/Library/CoreServices/SystemVersion.plist)))
   LION        = $(strip $(findstring 10.7, $(shell egrep "<string>10\.7" /System/Library/CoreServices/SystemVersion.plist)))
   MOUNTAIN    = $(strip $(findstring 10.8, $(shell egrep "<string>10\.8" /System/Library/CoreServices/SystemVersion.plist)))
   MAVERICKS   = $(strip $(findstring 10.9, $(shell egrep "<string>10\.9" /System/Library/CoreServices/SystemVersion.plist)))
endif 

ifeq ("$(OSUPPER)","LINUX")
     NVCC ?= $(CUDA_PATH)/bin/nvcc -ccbin $(CXX)
else
  ifneq ($(DARWIN),)
    # for some newer versions of XCode, CLANG is the default compiler, so we need to include this
    ifdef MAVERICKS
      NVCC   ?= $(CUDA_PATH)/bin/nvcc -ccbin $(CLANG)
      STDLIB ?= -stdlib=libstdc++
    else
      NVCC   ?= $(CUDA_PATH)/bin/nvcc -ccbin $(CXX)
    endif
  else
    $(error Fatal Error: This Makefile only works on Unix platforms.)
  endif
endif



#------------------------------------------------------------------------------
# Common Flags
#------------------------------------------------------------------------------

# nvcc compile flags
NVCCFL             := -m${OS_SIZE}
NVCCINCFL          := -I$(CUDA_PATH)/samples/common/inc
NVCCLDFL           :=

# gcc compile flags
CXXFL              :=
CXXSHRFL           :=
CXXINCFL           :=
CXXLIBFL           :=

# link flags
ifeq ($(OS_SIZE),32)
	NVCCLDFL       := -L$(CUDA_PATH)/lib -lcudart 
else
	NVCCLDFL       := -L$(CUDA_PATH)/lib64 -lcudart 
endif


# find NVCC version
NVCC_MAJOR_NUM     := $(shell nvcc -V 2>/dev/null | grep -o 'release [0-9]\.' | grep -o '[0-9]')
NVCCFL             += -D__CUDA$(NVCC_MAJOR_NUM)__

# CUDA code generation flags
GENCODE_SM20       := -gencode arch=compute_20,code=sm_20
GENCODE_SM30       := -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=\"sm_35,compute_35\"
NVCCFL             += $(GENCODE_SM20) $(GENCODE_SM30)
NVCCFL             += -Wno-deprecated-gpu-targets

# OS-specific build flags
ifneq ($(DARWIN),)
	CXXLIBFL       += -rpath $(CUDA_PATH)/lib
	CXXFL          += -arch $(OS_ARCH) $(STDLIB)  
else
	ifeq ($(OS_ARCH),armv7l)
		ifeq ($(abi),gnueabi)
			CXXFL     += -mfloat-abi=softfp
		else
			# default to gnueabihf
			override abi := gnueabihf
			CXXLIBFL  += --dynamic-linker=/lib/ld-linux-armhf.so.3
			CXXFL     += -mfloat-abi=hard
		endif
	endif
endif

# shared library flags
CXXSHRFL           += -fPIC -shared


#------------------------------------------------------------------------------
# CARLsim Library
#------------------------------------------------------------------------------

# variables starting with CARLSIM4_ are intended for the user
# variables starting with SIM_ are for internal use when installing CARLsim
SIM_LIB_NAME := carlsim
SIM_MAJOR_NUM := 4
SIM_MINOR_NUM := 0
SIM_BUILD_NUM := 0

lib_name := lib$(SIM_LIB_NAME).a
lib_ver := $(SIM_MAJOR_NUM).$(SIM_MINOR_NUM).$(SIM_BUILD_NUM)

output += $(lib_name) $(lib_name).$(lib_ver)


# use the following flags when building from CARLsim lib path
ifdef CARLSIM4_INSTALL_DIR
	CARLSIM4_INC_DIR  := $(CARLSIM4_INSTALL_DIR)/include
	CARLSIM4_LIB_DIR  := $(CARLSIM4_INSTALL_DIR)/lib
	sim_install_files += $(CARLSIM4_INC_DIR)
else
	CARLSIM4_INSTALL_DIR := $(HOME)/CARL
	CARLSIM4_INC_DIR  := $(CARLSIM4_INSTALL_DIR)/include
	CARLSIM4_LIB_DIR  := $(CARLSIM4_INSTALL_DIR)/lib
	sim_install_files += $(CARLSIM4_INC_DIR)
endif

sim_install_files += $(CARLSIM4_LIB_DIR)/$(lib_name)*

CARLSIM4_FLG := -I$(CARLSIM4_INC_DIR) -L$(CARLSIM4_LIB_DIR)
CARLSIM4_LIB := -l$(SIM_LIB_NAME)
