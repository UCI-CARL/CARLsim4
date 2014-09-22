# Makefile for building example program from the CARLsim source or library
example := random
output := *.dot *.dat *.log

# set absolute path to configured user.mk file or just copy it to this
# directory and leave USER_MK_PATH blank.
USER_MK_PATH = /home/kris/Project/CARLsim/
include $(USER_MK_PATH)user.mk

# TODO: make it so that this doesn't have to be included in the directory.
# It should be included relative to the library install path or something.
# It could also be rolled into user.mk.
include $(CARLSIM_SRC_DIR)/carlsim/carlsim.mk
# local info (vars can be overwritten)
local_src  := main_$(example).cpp
local_prog := $(example)

.PHONY: clean distclean
# compile from lib case
ifeq (${USE_CARLSIM_SRC},0)
output_files += $(local_prog)
$(local_prog): $(local_src)
	$(NVCC) $(CARLSIM_INCLUDES) $(CARLSIM_FLAGS) $(CARLSIM_LFLAGS) \
	$(CARLSIM_LIBS) $< -o $@
# compile from src case
else
output_files += $(local_prog) $(carlsim_objs)
$(local_prog): $(local_src) $(carlsim_sources) $(carlsim_objs)
	$(NVCC) $(CARLSIM_INCLUDES) $(CARLSIM_FLAGS) $(CARLSIM_LFLAGS) \
	$(CARLSIM_LIBS) $(carlsim_objs) $< -o $@
endif

clean:
	$(RM) $(output_files)

# TODO: distclean
distclean:
	$(RM) $(output_files) *.log *.dat results/*.dat

devtest:
	@echo $(output_files)
