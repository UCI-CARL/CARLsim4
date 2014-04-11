# CARLsim flags for core

#---------------------------------------------------------------------------
# CARLsim analysis class local variables
#---------------------------------------------------------------------------
local_dir := $(spike_info_dir)
local_deps := $(addprefix $(local_dir)/, spike_info.h \
	spike_info.cpp)
local_src := $(addprefix $(local_dir)/, spike_info.cpp)
local_objs := $(addprefix $(local_dir)/,spike_info.o)

sources += $(local_src)
objects += $(local_objs)
all_targets += spike_info
spike_info_ext_dir := $(interface_dir)/include
spike_info_ext_deps := $(spike_info_ext_dir)/carlsim_datastructures.h
spike_info_objs := $(local_objs)
spike_info_flags := -I$(spike_info_dir)
#---------------------------------------------------------------------------
# CARLsim spike_info class recipe
#---------------------------------------------------------------------------
.PHONY: spike_info
spike_info: $(local_deps) $(local_objs) $(spike_info_ext_deps)

# spike_info object
$(spike_info_dir)/%.o: $(spike_info_dir)/%.cpp $(local_deps)
	$(CXX) -c $(CPPFLAGS) -I$(spike_info_ext_dir) $< -o $@
