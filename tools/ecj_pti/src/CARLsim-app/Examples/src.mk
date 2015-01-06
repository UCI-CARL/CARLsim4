# module include file for example programs

#local info (vars can be overwritten)
local_dir := $(ex_dir)

# Examples that don't have any special dependencies.
example_names := ReprintExample SumExample
example := $(addprefix $(local_dir)/, $(example_names))
output := error.log
local_prog := $(example)
local_objs := $(addsuffix .o, $(example))

# Examples that will need their own special target
special_examples := IzkExample SimpleCA3 TuneFiringRatesECJ

# pass these to the Makefile
sources += $(local_src)
output_files += $(output)
objects += $(local_objs)
pti_programs += $(local_prog) $(addprefix $(local_dir)/, $(special_examples))
all_targets += $(pti_programs)

# carlsim information
carlsim_main := $(CARLSIM_SRC_DIR)/carlsim
carlsim_includes := -I$(carlsim_main)/kernel/include/ -I$(carlsim_main)/interface/include/ -I$(carlsim_main)/spike_monitor/
carlsim_local_lib := libCARLsim.a
carlsim_lib := $(addprefix $(carlsim_main)/,$(carlsim_local_lib))
libraries += $(carlsim_lib) $(carlsim_main)/*.a.*

# Rules for example binaries that use EO/PTI
.PHONY: $(example_names) $(special_examples)
$(local_dir)/%.o: $(local_dir)/%.cpp $(pti_deps)
	$(CC) -c $(PTI_FLAGS) $< -o $@
$(local_prog): %: %.o $(pti_deps) $(pti_objs)
	$(CC) $(PTI_FLAGS) $< $(pti_objs) -o $@ $(LDFLAGS)

izk_lib := $(izk_dir)/libizk.a
$(local_dir)/IzkExample: $(local_dir)/IzkExample.cpp $(pti_deps) $(pti_objs)
	$(MAKE) -C $(izk_dir)/ libizk.a
	$(CC) -g $(PTI_FLAGS) $< $(pti_objs) $(izk_lib) -o $@ $(LDFLAGS)

$(local_dir)/SimpleCA3: $(local_dir)/SimpleCA3.cpp $(pti_deps) $(pti_objs)
	$(MAKE) -C $(CARLSIM_SRC_DIR) libCARLsim
	@cp $(CARLSIM_SRC_DIR)/carlsim/libCARLsim*.*.* $(CARLSIM_SRC_DIR)/carlsim/libCARLsim.a
	nvcc -g $(PTI_FLAGS) $(carlsim_includes) $< $(pti_objs) $(carlsim_lib) -o $@ $(LDFLAGS)

$(local_dir)/TuneFiringRatesECJ: $(local_dir)/TuneFiringRatesECJ.cpp $(pti_deps) $(pti_objs)
	$(MAKE) -C $(CARLSIM_SRC_DIR) libCARLsim
	@cp $(CARLSIM_SRC_DIR)/carlsim/libCARLsim*.*.* $(CARLSIM_SRC_DIR)/carlsim/libCARLsim.a
	nvcc -g $(PTI_FLAGS) $(carlsim_includes) $(CARLSIM_LFLAGS) $(CARLSIM_FLAGS) \
		$< $(pti_objs) $(carlsim_lib) -o $@ $(LDFLAGS)

# these make it so you can type 'make <example_name>' with tab-complete
ReprintExample: $(local_dir)/ReprintExample

SumExample: $(local_dir)/SumExample

IzkExample: $(local_dir)/IzkExample

SimpleCA3: $(local_dir)/SimpleCA3

TuneFiringRatesECJ: $(local_dir)/TuneFiringRatesECJ
