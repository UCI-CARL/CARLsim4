#-------------------------------------------------------------------------------
# set user-defined flags
#-------------------------------------------------------------------------------
# common flags
CXX = g++
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
