# Makefile for building project program from the CARLsim library

# NOTE: if you are compiling your code in a directory different from
# examples/<example_name> or projects/<project_name> then you need to either
# move the configured user.mk file to this directory or set the path to
# where CARLsim can find the user.mk.
USER_MK_PATH = ../../carlsim/
include $(USER_MK_PATH)configure.mk

project := hello_world
output := *.dot *.dat *.log *.csv

# You should not need to edit the file beyond this point
# ------------------------------------------------------

# we are compiling from lib
CARLSIM_FLAGS += -I$(CARLSIM_INSTALL_DIR)/inc
CARLSIM_LIBS  += -L$(CARLSIM_INSTALL_DIR)/lib -lCARLsim -lcurand

local_src  := main_$(project).cpp
local_prog := $(project)

# you can add your own local objects
local_objs :=

output_files += $(local_prog) $(local_objs)

.PHONY: clean distclean
# compile from CARLsim lib
$(local_prog): $(local_src) $(local_objs)
	$(NVCC) $(NVCCSHRFL) $(NVCCINCFL) $(CARLSIM_FLAGS) $(NVCCFL) $< -o $@ $(CARLSIM_LIBS)

clean:
	$(RM) $(output_files)

distclean:
	$(RM) $(output_files) results/*

devtest: @echo $(CARLSIM_FLAGS)
