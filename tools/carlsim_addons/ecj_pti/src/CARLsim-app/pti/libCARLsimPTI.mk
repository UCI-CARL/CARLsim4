num_ver := $(pti_major_num).$(pti_minor_num)

lib_ver := $(num_ver).$(pti_build_num)

lib_name := libCARLsimPTI.a

pti_lib := $(pti_dir)/$(lib_name)
# keep track of this so we can delete it later on distclean
libraries += $(pti_lib).$(lib_ver)

# TODO: I should remove this. I should be all_targets anyway
#default_targets += libCARLsim
install_targets += install_carlsim_pti
.PHONY: libCARLsimPTI install_carlsim_pti
libCARLsimPTI: $(pti_lib)

$(pti_lib): $(pti_sources) $(pti_inc) $(pti_objs)
	ar rcs $@.$(lib_ver) $(pti_objs)

install_carlsim_pti: $(pti_lib)
	@test -d $(ECJ_PTI_DIR) || mkdir -p $(ECJ_PTI_DIR)
	@test -d $(ECJ_PTI_DIR)/lib || mkdir $(ECJ_PTI_DIR)/lib
	@test -d $(ECJ_PTI_DIR)/include || mkdir $(ECJ_PTI_DIR)/include
	@install -m 0755 $(pti_lib).$(lib_ver) $(ECJ_PTI_DIR)/lib
	@ln -Tfs $(ECJ_PTI_DIR)/lib/$(lib_name).$(lib_ver) \
		$(ECJ_PTI_DIR)/lib/$(lib_name).$(num_ver)
	@ln -Tfs $(ECJ_PTI_DIR)/lib/$(lib_name).$(num_ver) \
		$(ECJ_PTI_DIR)/lib/$(lib_name)
	@install -m 0644 $(pti_dir)/Experiment.h $(pti_dir)/Logger.h \
		$(pti_dir)/PTI.h $(pti_dir)/Util.h $(pti_dir)/ParameterInstances.h \
		$(ECJ_PTI_DIR)/include
