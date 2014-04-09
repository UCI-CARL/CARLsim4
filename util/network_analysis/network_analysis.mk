# CARLsim flags for core

#---------------------------------------------------------------------------
# CARLsim analysis class local variables
#---------------------------------------------------------------------------
local_dir := $(net_analysis_dir)
local_deps := $(addprefix $(local_dir)/, network_analysis.h \
	network_analysis.cpp)
local_src := $(addprefix $(local_dir)/, network_analysis.cpp)
local_objs := $(addprefix $(local_dir)/,network_analysis.o)

sources += $(local_src)
objects += $(local_objs)
all_targets += network_analysis
net_analysis_ext_dir := $(interface_dir)/include
net_analysis_ext_deps := $(net_analysis_ext_dir)/carlsim_datastructures.h
net_analysis_objs := $(local_objs)
net_analysis_flags := -I$(net_analysis_dir)
#---------------------------------------------------------------------------
# CARLsim network_analysis class recipe
#---------------------------------------------------------------------------
.PHONY: network_analysis
network_analysis: $(local_deps) $(local_objs) $(net_analysis_ext_deps)

# network_analysis object
$(net_analysis_dir)/%.o: $(net_analysis_dir)/%.cpp $(local_deps)
	$(CXX) -c $(CPPFLAGS) -I$(net_analysis_ext_dir) $< -o $@
