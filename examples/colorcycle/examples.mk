# module include file for simple
example := colorcycle
output := *.dot *.txt *.log 

# local info (vars can be overwritten)
local_dir := examples/$(example)
local_src := $(local_dir)/main_$(example).cpp
local_objs := examples/common/v1ColorME.2.0.o
local_prog := $(local_dir)/$(example)

# info passed up to Makfile
sources += $(local_src)
carlsim_programs += $(local_prog)
output_files += $(addprefix $(local_dir)/,$(output))
all_targets += $(local_prog)
objects += $(local_objs)

colorcycle_objs := $(local_objs)

.PHONY: $(example)
$(example): $(local_src) $(local_prog)

$(local_prog): $(local_src) $ $(carlsim_deps) $(carlsim_objs) $(local_objs)
	$(NVCC) $(CARLSIM_INCLUDES) $(CARLSIM_LFLAGS) $(CARLSIM_LIBS) \
	$(CARLSIM_FLAGS) $(carlsim_objs) $(colorcycle_objs) $< -o $@

# local cuda
examples/common/%.o: examples/common/%.cu $(local_deps)
	$(NVCC) -c $(CARLSIM_INCLUDES) $(CARLSIM_FLAGS) $< -o $@