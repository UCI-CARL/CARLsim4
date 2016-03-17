#------------------------------------------------------------------------------
# libCARLsim main makefile
#
# Note: This file depends on variables set in user.mk, settings.mk, and
# carlsim.mk, thus must be run after importing those others.
#------------------------------------------------------------------------------

CARLSIM_MAJOR_NUM := 4
CARLSIM_MINOR_NUM := 0
CARLSIM_BUILD_NUM := 0

lib_ver := $(CARLSIM_MAJOR_NUM).$(CARLSIM_MINOR_NUM).$(CARLSIM_BUILD_NUM)
lib_name := libCARLsim.a

targets += $(lib_name)
libraries += $(lib_name).$(lib_ver)

.PHONY: $(lib_name) install

install: $(lib_name)
	ar rcs $(lib_name).$(lib_ver) $(intf_obj_files) $(krnl_obj_files) $(grps_obj_files) $(spks_obj_files) $(conn_obj_files)
	@test -d $(CARLSIM_INSTALL_DIR) || mkdir -p $(CARLSIM_INSTALL_DIR)
	@test -d $(CARLSIM_INSTALL_DIR)/lib || mkdir $(CARLSIM_INSTALL_DIR)/lib
	@test -d $(CARLSIM_INSTALL_DIR)/inc || mkdir $(CARLSIM_INSTALL_DIR)/inc
	@install -m 0755 $(lib_name).$(lib_ver) $(CARLSIM_INSTALL_DIR)/lib
	@ln -Tfs $(CARLSIM_INSTALL_DIR)/lib/$(lib_name).$(lib_ver) $(CARLSIM_INSTALL_DIR)/lib/$(lib_name).$(CARLSIM_MAJOR_NUM).$(CARLSIM_MINOR_NUM)
	@ln -Tfs $(CARLSIM_INSTALL_DIR)/lib/$(lib_name).$(CARLSIM_MAJOR_NUM).$(CARLSIM_MINOR_NUM) $(CARLSIM_INSTALL_DIR)/lib/$(lib_name)
	@install -m 0644 $(intf_inc_files) $(CARLSIM_INSTALL_DIR)/inc
	@install -m 0644 $(krnl_inc_files) $(CARLSIM_INSTALL_DIR)/inc
	@install -m 0644 $(grps_inc_files) $(CARLSIM_INSTALL_DIR)/inc
	@install -m 0644 $(conn_inc_files) $(CARLSIM_INSTALL_DIR)/inc
	@install -m 0644 $(spks_inc_files) $(CARLSIM_INSTALL_DIR)/inc

# install_backends: $(back_so_files)
# 	@install -m 0755 $(back_so_files) $(CARLSIM_INSTALL_DIR)/lib