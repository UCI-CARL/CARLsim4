# module include file for tuneFiringRates example program
example := spikeInfo
output := *.o results/EA-Data results/*.txt

#local info (vars can be overwritten)
local_dir := examples/$(example)
local_src := $(local_dir)/main_$(example).cpp
local_prog := $(local_dir)/$(example)
local_objs := $(local_dir)/main_$(example).o

# pass these to the Makefile
sources += $(local_src)
output_files += $(addprefix $(local_dir)/,$(output))
objects += $(local_objs)
pti_programs += $(local_prog)

# Rules for example binaries that use EO/PTI
.PHONY: $(example)
$(example): $(local_src) $(local_prog)

$(local_prog): $(local_src) $(local_objs) $(carlsim_sources) \
	$(carlsim_objs) $(spike_monitor_obs)
	$(NVCC) $(CARLSIM_INCLUDES) $(CARLSIM_LFLAGS) $(CARLSIM_FLAGS) \
	$(EO_FLAGS) $(EO_LFLAGS) $(PTI_FLAGS) $(PTI_LFLAGS) $(spike_monitor_flags) \
	$< $(carlsim_objs) -o $@ $(CARLSIM_LIBS) $(PTI_LIBS) $(EO_LIBS)

$(local_objs): $(local_src)
	$(NVCC) -c $(CARLSIM_INCLUDES) $(CARLSIM_FLAGS) $(PTI_FLAGS) $(EO_FLAGS) \
	$(spike_monitor_flags) $< -o $@

