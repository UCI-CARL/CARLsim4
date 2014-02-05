#-------------------------------------------------------------------------------
# set user-defined flags
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

# set debug flag
ifeq (${strip ${CARLSIM_DEBUG}},1)
	DEBUG_FLAG = -g
else
	DEBUG_FLAG = 
endif

# common flags
CC = g++
NVCC = nvcc
CPPFLAGS = $(DEBUG_FLAG) $(OPT_FLAG) -Wall -std=c++0x

MV := mv -f
RM := rm -rf

output_files += *.dot *.txt *.log tmp* *.status

help:
	@echo -e '\n'Type \'make\' or \'make all\' to make CARLsim and CARLsim \
	examples.'\n'
	@echo -e Type \'make pti\' to make the pti library, install it, \
	and make the pti examples.'\n'
	@echo -e Type \'make uninstall\' to uninstall the pti library.'\n'
	@echo -e To compile a specific example, type \'make \<example folder \
	name\>\'.'\n'
	@echo -e Note: simpleEA, tuneFiringRates, and SORFs examples \
	require CARLsim PTI installation.'\n'
