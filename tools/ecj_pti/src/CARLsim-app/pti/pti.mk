# module include file for CARLsim pti
# edited by KDC

#-------------------------------------------------------------------------------
# Release information
#-------------------------------------------------------------------------------
pti_major_num := 2
pti_minor_num := 0
pti_build_num := 0

#-------------------------------------------------------------------------------
# CARLsim pti flags
#-------------------------------------------------------------------------------

#TODO: CARLSIM_FASTMATH and CARLSIM_CUOPTLEVEL should be just FASTMATH AND OPTLEVEL
#TODO: Compile as library
#TODO: consider install/uninstall targets

# use fast math
ifeq (${strip ${CARLSIM_FASTMATH}},1)
	PTI_FLAGS += -use_fast_math
endif

# append include path to PTI_FLAGS
PTI_FLAGS += -I$(CURDIR)/$(pti_dir)

#-------------------------------------------------------------------------------
# CARLsim pti local variables
#-------------------------------------------------------------------------------
local_dir := $(pti_dir)
local_deps := Logger.cpp PTI.cpp ParameterInstances.cpp Util.cpp
local_src := $(addprefix $(local_dir)/,$(local_deps))
local_objs := $(addprefix $(local_dir)/, Logger.o PTI.o ParameterInstances.o Util.o)
local_dbg += $(local_src:.cpp=.gcno)
local_dbg += $(local_src:.cpp=.gcda)

sources += $(local_src)
pti_deps += $(local_deps)
pti_objs += $(local_objs)
pti_sources += $(local_src)
objects += $(pti_objs)
all_targets += pti
output_files += debug.log $(local_dbg)

#-------------------------------------------------------------------------------
# CARLsim pti rules
#-------------------------------------------------------------------------------
.PHONY: pti
pti: $(local_objs)

$(local_dir)/%.o: $(local_dir)/%.cpp $(local_deps)
	$(CC) $(CPPFLAGS) $(DL_FLAGS) -c $(PTI_FLAGS) $< -o $@

#$(local_dir)/lib%.so: $(local_dir)/%.o
#	$(CC) $(CPPFLAGS) $(DL_FLAGS) -o \
#	$@.$(pti_major_ver).$(pti_minor_ver).$(pti_rel_ver) $<; \
#	ln -fs lib$*.so.$(pti_major_ver).$(pti_minor_ver).$(pti_rel_ver) \
#	$@.$(pti_major_ver).$(pti_minor_ver); \
#	ln -fs lib$*.so.$(pti_major_ver).$(pti_minor_ver) $@
