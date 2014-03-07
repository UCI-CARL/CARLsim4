# module include file for tuneFiringRates example program
example := tuneFiringRates
output := *.dot *.txt *.log tmp* *.status *.o

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

$(local_prog): $(local_src) $(local_objs) $(carlsim_sources) $(carlsim_objs) \
	$(common_sources) $(common_objs)
	$(NVCC) $(CARLSIM_INCLUDES) $(CARLSIM_LFLAGS) $(CARLSIM_FLAGS) \
	$(EO_FLAGS) $(PTI_FLAGS) $< $(carlsim_objs) $(common_objs) -o $@ \
	$(CARLSIM_LIBS) $(PTI_LIBS) $(EO_LIBS)

$(local_objs): $(local_src)
	$(NVCC) -c $(CARLSIM_INCLUDES) $(CARLSIM_LFLAGS) $(CARLSIM_FLAGS) \
	$(PTI_FLAGS) $(EO_FLAGS) $< -o $@ $(CARLSIM_LIBS) $(PTI_LIBS) $(EO_LIBS)
