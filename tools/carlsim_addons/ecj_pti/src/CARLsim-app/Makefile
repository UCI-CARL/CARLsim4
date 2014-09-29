# Makefile to build the pti class
#-------------------------------------------------------------------------------
# Begin user modifiable section
#-------------------------------------------------------------------------------
# if optional env vars do not exist, assign default values
# $(OPT_LEVEL): set to 1, 2, or 3 if you want to use optimization.  Default: 0.
# $(DEBUG_INFO): set to 1 to include debug info, set to 0 to not include 
# debugging info.  Default: 0.
CARLSIM_CUDAVER ?= 3
CARLSIM_FASTMATH ?= 0
CARLSIM_CUOPTLEVEL ?= 0
CARLSIM_DEBUG ?= 0
CARLSIM_ROOT ?= $(HOME)/Project/CARLsim/
#-------------------------------------------------------------------------------
# End user modifiable section
#-------------------------------------------------------------------------------

# set optimization flag
ifeq (${strip ${CARLSIM_CUOPTLEVEL}},0)
	OPT_FLAG = 
else ifeq (${strip ${OPT_LEVEL}},1)
	OPT_FLAG = -O1
else ifeq (${strip ${OPT_LEVEL}},2)
	OPT_FLAG = -O2
else ifeq (${strip ${OPT_LEVEL}},3)
	OPT_FLAG = -O3
else
	OPT_FLAG = 
endif

ifeq (${strip ${CARLSIM_CUDAVER}},3)
	CARLSIM_LFLAGS = -L${NVIDIA_SDK}/C/lib
	CARLSIM_LIBS = -lcutil_x86_64
else
	CARLSIM_LFLAGS =
	CARLSIM_LIBS =
endif

# set debug flag
ifeq (${strip ${CARLSIM_DEBUG}},1)
	DEBUG_FLAG = -g
else
	DEBUG_FLAG =
endif
# common flags
CC = g++
NVCC = nvcc
CPPFLAGS = $(DEBUG_FLAG) $(OPT_FLAG) -Wall --coverage
LDFLAGS = -lgcov

all_targets :=
all_projects :=
carlsim_programs  := 
pti_programs :=
pti_deps :=
pti_objs :=
sources   :=
libraries := 
carlsim_deps :=
carlsim_sources :=
carlsim_objs := 
common_sources :=
common_objs :=
output_files :=
objects :=
izk_build_files :=
pti_dir = pti

# location of .cpp files
vpath %.cpp $(pti_dir)
# location of .h files
vpath %.h $(pti_dir)

all:

include pti/pti.mk
include Examples/examples.mk
include ../izk/izk.mk

all: $(all_targets)

clean:
	$(RM) $(objects) $(carlsim_programs) $(pti_programs) $(output_files) $(izk_build_files)

