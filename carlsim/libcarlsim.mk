num_ver := $(carlsim_major_num).$(carlsim_minor_num)

lib_ver := $(num_ver).$(carlsim_build_num)

lib_name := libCARLsim.a

carlsim_lib := $(addprefix carlsim/,$(lib_name))
# keep track of this so we can delete it later on distclean
libraries += $(carlsim_lib).$(lib_ver)

default_targets += libCARLsim
.PHONY: libCARLsim install
libCARLsim: $(carlsim_lib)

$(carlsim_lib): $(carlsim_sources) $(carlsim_inc) $(carlsim_objs)
	ar rcs $@.$(lib_ver) $(carlsim_objs)

install: $(carlsim_lib)
	@test -d $(CARLSIM_LIB_DIR) || \
		mkdir -p $(CARLSIM_LIB_DIR)
	@test -d $(CARLSIM_LIB_DIR)/lib || mkdir \
		$(CARLSIM_LIB_DIR)/lib
	@test -d $(CARLSIM_LIB_DIR)/include || mkdir \
		$(CARLSIM_LIB_DIR)/include
	@test -d $(CARLSIM_LIB_DIR)/include/kernel || mkdir \
		$(CARLSIM_LIB_DIR)/include/kernel
	@test -d $(CARLSIM_LIB_DIR)/include/interface || mkdir \
		$(CARLSIM_LIB_DIR)/include/interface
	@test -d $(CARLSIM_LIB_DIR)/include/connection_monitor || mkdir \
		$(CARLSIM_LIB_DIR)/include/connection_monitor
	@test -d $(CARLSIM_LIB_DIR)/include/spike_monitor || mkdir \
		$(CARLSIM_LIB_DIR)/include/spike_monitor
	@test -d $(CARLSIM_LIB_DIR)/include/group_monitor || mkdir \
		$(CARLSIM_LIB_DIR)/include/group_monitor
	@test -d $(CARLSIM_LIB_DIR)/include/spike_generators || mkdir \
		$(CARLSIM_LIB_DIR)/include/spike_generators
	@test -d $(CARLSIM_LIB_DIR)/include/simple_weight_tuner || mkdir \
		$(CARLSIM_LIB_DIR)/include/simple_weight_tuner \
	@test -d $(CARLSIM_LIB_DIR)/include/stopwatch || mkdir \
		$(CARLSIM_LIB_DIR)/include/stopwatch
	@test -d $(CARLSIM_LIB_DIR)/include/visual_stimulus || mkdir \
		$(CARLSIM_LIB_DIR)/include/visual_stimulus
	@install -m 0755 $(carlsim_lib).$(lib_ver) $(CARLSIM_LIB_DIR)/lib
	@ln -Tfs $(CARLSIM_LIB_DIR)/lib/$(lib_name).$(lib_ver) \
		$(CARLSIM_LIB_DIR)/lib/$(lib_name).$(num_ver)
	@ln -Tfs $(CARLSIM_LIB_DIR)/lib/$(lib_name).$(num_ver) \
		$(CARLSIM_LIB_DIR)/lib/$(lib_name)
	@install -m 0644 $(kernel_dir)/include/cuda_version_control.h \
		$(kernel_dir)/include/snn_definitions.h \
		$(CARLSIM_LIB_DIR)/include/kernel
	@install -m 0644 $(interface_dir)/include/callback.h \
		$(interface_dir)/include/carlsim_datastructures.h \
		$(interface_dir)/include/carlsim_definitions.h \
		$(interface_dir)/include/carlsim_log_definitions.h \
		$(interface_dir)/include/linear_algebra.h \
		$(interface_dir)/include/poisson_rate.h \
		$(interface_dir)/include/carlsim.h $(interface_dir)/include/user_errors.h \
		$(CARLSIM_LIB_DIR)/include/interface
	@install -m 0644 $(conn_mon_dir)/connection_monitor.h \
		$(CARLSIM_LIB_DIR)/include/connection_monitor
	@install -m 0644 $(spike_mon_dir)/spike_monitor.h \
		$(CARLSIM_LIB_DIR)/include/spike_monitor
	@install -m 0644 $(group_mon_dir)/group_monitor.h \
		$(CARLSIM_LIB_DIR)/include/group_monitor
	@install -m 0644 $(tools_visualstim_dir)/visual_stimulus.h \
		$(CARLSIM_LIB_DIR)/include/visual_stimulus
	@install -m 0644 $(tools_spikegen_dir)/periodic_spikegen.h \
		$(tools_spikegen_dir)/spikegen_from_file.h \
		$(tools_spikegen_dir)/spikegen_from_vector.h \
		$(tools_spikegen_dir)/interactive_spikegen.h \
		$(tools_spikegen_dir)/pre_post_group_spikegen.h \
		$(CARLSIM_LIB_DIR)/include/spike_generators
	@install -m 0644 $(tools_swt_dir)/simple_weight_tuner.h \
		$(CARLSIM_LIB_DIR)/include/simple_weight_tuner
	@install -m 0644 $(tools_stopwatch_dir)/stopwatch.h \
		$(CARLSIM_LIB_DIR)/include/stopwatch

# uninstall LIB folder, which by default is under /opt/
uninstall:
	@test -d $(CARLSIM_LIB_DIR) && $(RM) $(CARLSIM_LIB_DIR)
