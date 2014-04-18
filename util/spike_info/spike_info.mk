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
spike_info_ext_deps := $(interface_dir)/include/carlsim_datastructures.h \
	$(carlsim_dir)/include/snn.h
spike_info_objs := $(local_objs)
spike_info_flags := -I$(spike_info_dir) -I$(interface_dir)/include \
	-I$(carlsim_dir)/include
#---------------------------------------------------------------------------
# CARLsim spike_info class recipe
#---------------------------------------------------------------------------
.PHONY: spike_info
spike_info: $(local_deps) $(local_objs) $(spike_info_ext_deps)

# spike_info object
$(spike_info_dir)/%.o: $(spike_info_dir)/%.cpp $(local_deps)
	$(CXX) -c $(CPPFLAGS) $(spike_info_flags) $< -o $@
