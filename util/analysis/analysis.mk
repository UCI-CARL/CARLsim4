# CARLsim flags for core

#---------------------------------------------------------------------------
# CARLsim analysis class local variables
#---------------------------------------------------------------------------
local_dir := $(analysis_dir)
local_deps := $(addprefix $(local_dir)/include/, analysis.h)
local_src := $(addprefix $(local_dir)/src/, analysis.cpp \
	$(local_deps))
local_objs := $(addprefix $(local_dir)/src/,analysis.o)

sources += $(local_src)
objects += $(local_objs)
all_targets += analysis

#---------------------------------------------------------------------------
# CARLsim analysis class recipe
#---------------------------------------------------------------------------
.PHONY: analysis
analysis: $(local_src) $(local_objs) $(carlsim_sources) $(carlsim_objs) \
	$(interface_src) $(interface_obj)

# analysis object
$(analysis_dir)/src/%.o: $(analysis_dir)/src/%.cpp $(local_deps)
	$(NVCC) -c $(CARLSIM_INCLUDES) $(CARLSIM_LFLAGS) $(CARLSIM_LIBS) \
	$(CARLSIM_FLAGS) $< -o $@
