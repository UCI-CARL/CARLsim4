# module include file for CUBA_GPU project program
project := CUBA_GPU
output := *.dot *.txt *.log

# local info (vars can be overwritten)
local_dir := projects/$(project)
local_src := $(local_dir)/main_$(project).cpp
local_prog := $(local_dir)/$(project)

# info passed up to Makefile
sources += $(local_src)
carlsim_programs += $(local_prog)
output_files += $(addprefix $(local_dir)/,$(output))
all_targets += $(local_prog)

.PHONY: $(project)
$(project): $(local_src) $(local_prog)

$(local_prog): $(local_src) $(carlsim_deps) $(carlsim_objs)
	$(NVCC) $(CARLSIM_INCLUDES) $(CARLSIM_FLAGS) $(CARLSIM_LFLAGS) \
	$(CARLSIM_LIBS) $(carlsim_objs) $< -o $@

