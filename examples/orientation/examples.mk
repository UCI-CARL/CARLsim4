# module include file for simple
example := orientation
output := *.dot *.txt *.log 

# local info (vars can be overwritten)
local_dir := examples/$(example)
local_src := $(local_dir)/main_$(example).cpp
local_objs := examples/common/v1ColorME.2.0.o
local_prog := $(local_dir)/$(example)
local_deps := $(local_src)

# utilities used
util_src := $(util_dir)/input_stimulus/input_stimulus.cpp
local_deps += $(util_dir)/input_stimulus/input_stimulus.h $(util_src)
CARLSIM_INCLUDES += -I$(CURDIR)/$(util_dir)/input_stimulus

# info passed up to Makfile
sources += $(local_src)
carlsim_programs += $(local_prog)
output_files += $(addprefix $(local_dir)/,$(output))
all_targets += $(local_prog)
objects += $(local_objs)

orientation_objs := $(local_objs)

.PHONY: $(example)
$(example): $(local_src) $(local_prog)

$(local_prog): $(local_deps) $(carlsim_deps) $(carlsim_objs) $(local_objs)
	$(NVCC) $(CARLSIM_INCLUDES) $(CARLSIM_LFLAGS) $(CARLSIM_LIBS) \
	$(CARLSIM_FLAGS) $(carlsim_objs) $(local_objs) $(util_src) $< -o $@

# local cuda
examples/common/%.o: examples/common/%.cu $(local_deps)
	$(NVCC) -c $(CARLSIM_INCLUDES) $(CARLSIM_FLAGS) $< -o $@