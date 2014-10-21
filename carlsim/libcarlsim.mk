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
	@test -d $(CARLSIM_LIB_DIR)/include/spike_monitor || mkdir \
		$(CARLSIM_LIB_DIR)/include/spike_monitor
	@test -d $(CARLSIM_LIB_DIR)/include/spike_generators || mkdir \
		$(CARLSIM_LIB_DIR)/include/spike_generators
	@install -m 0755 $(carlsim_lib).$(lib_ver) $(CARLSIM_LIB_DIR)/lib
	@ln -Tfs $(CARLSIM_LIB_DIR)/lib/$(lib_name).$(lib_ver) \
		$(CARLSIM_LIB_DIR)/lib/$(lib_name).$(num_ver)
	@ln -Tfs $(CARLSIM_LIB_DIR)/lib/$(lib_name).$(num_ver) \
		$(CARLSIM_LIB_DIR)/lib/$(lib_name)
	@install -m 0644 $(kernel_dir)/include/cuda_version_control.h \
		$(kernel_dir)/include/poisson_rate.h $(kernel_dir)/include/mtrand.h \
		$(CARLSIM_LIB_DIR)/include/kernel
	@install -m 0644 $(interface_dir)/include/callback.h \
		$(interface_dir)/include/carlsim_datastructures.h \
		$(interface_dir)/include/carlsim_definitions.h \
		$(interface_dir)/include/linear_algebra.h \
		$(interface_dir)/include/carlsim.h $(interface_dir)/include/user_errors.h \
		$(CARLSIM_LIB_DIR)/include/interface
	@install -m 0644 $(spike_mon_dir)/spike_monitor.h \
	$(CARLSIM_LIB_DIR)/include/spike_monitor
	@install -m 0644 $(spike_gen_dir)/periodic_spikegen.h \
		$(spike_gen_dir)/spikegen_from_file.h \
		$(spike_gen_dir)/spikegen_from_vector.h $(CARLSIM_LIB_DIR)/include/spike_generators
