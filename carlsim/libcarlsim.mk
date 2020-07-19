##----------------------------------------------------------------------------##
##
##   CARLsim5 Library
##   ----------------
##
##   Authors:   Michael Beyeler <mbeyeler@uci.edu>
##              Kristofor Carlson <kdcarlso@uci.edu>
##
##   Institute: Cognitive Anteater Robotics Lab (CARL)
##              Department of Cognitive Sciences
##              University of California, Irvine
##              Irvine, CA, 92697-5100, USA
##
##   Version:   03/31/2016
##
##----------------------------------------------------------------------------##

#------------------------------------------------------------------------------
# CARLsim5 Library Targets
#------------------------------------------------------------------------------

.PHONY: $(lib_name) test_env install uninstall
.PHONY: create_files welcome uninstall delete_files farewell

install: test_env create_files welcome

uninstall: test_env delete_files farewell


test_env:
ifndef CARLSIM5_LIB_DIR
	$(error CARLSIM5_LIB_DIR not set. Run with -E: $$ sudo -E make install)
else
	$(info CARLsim5 library path: $(CARLSIM5_LIB_DIR))
endif
ifndef CARLSIM5_INC_DIR
	$(error CARLSIM5_INC_DIR not set. Run with -E: $$ sudo -E make install)
else
	$(info CARLsim5 include path: $(CARLSIM5_INC_DIR))
endif

create_files:
ifdef CARLSIM5_INSTALL_DIR
	@test -d $(CARLSIM5_INSTALL_DIR) || mkdir $(CARLSIM5_INSTALL_DIR)
endif
	@test -d $(CARLSIM5_INC_DIR) || mkdir $(CARLSIM5_INC_DIR)
	@test -d $(CARLSIM5_LIB_DIR) || mkdir $(CARLSIM5_LIB_DIR)
	@install -m 0755 $(lib_name).$(lib_ver) $(CARLSIM5_LIB_DIR)
ifneq ($(DARWIN),)
	@ln -fs $(CARLSIM5_LIB_DIR)/$(lib_name).$(lib_ver) $(CARLSIM5_LIB_DIR)/$(lib_name).$(SIM_MAJOR_NUM).$(SIM_MINOR_NUM)
	@ln -fs $(CARLSIM5_LIB_DIR)/$(lib_name).$(SIM_MAJOR_NUM).$(SIM_MINOR_NUM) $(CARLSIM5_LIB_DIR)/$(lib_name)
else
	@ln -Tfs $(CARLSIM5_LIB_DIR)/$(lib_name).$(lib_ver) $(CARLSIM5_LIB_DIR)/$(lib_name).$(SIM_MAJOR_NUM).$(SIM_MINOR_NUM)
	@ln -Tfs $(CARLSIM5_LIB_DIR)/$(lib_name).$(SIM_MAJOR_NUM).$(SIM_MINOR_NUM) $(CARLSIM5_LIB_DIR)/$(lib_name)
endif
	@install -m 0644 $(intf_inc_files) $(CARLSIM5_INC_DIR)
	@install -m 0644 $(krnl_inc_files) $(CARLSIM5_INC_DIR)
	@install -m 0644 $(mon_inc_files) $(CARLSIM5_INC_DIR)
	@install -m 0644 $(swt_inc_files) $(CARLSIM5_INC_DIR)
	@install -m 0644 $(spkgen_inc_files) $(CARLSIM5_INC_DIR)
	@install -m 0644 $(stp_inc_files) $(CARLSIM5_INC_DIR)
	@install -m 0644 $(vs_inc_files) $(CARLSIM5_INC_DIR)
	@install -m 0644 $(add_files) $(CARLSIM5_INC_DIR)

delete_files: test_env
ifeq (,$(findstring $(lib_name),$(sim_install_files)))
	$(error Something went wrong. None of the files that are about to be deleted contain the string "$(lib_name)". Delete files manually)
endif
	$(RMR) $(sim_install_files)

welcome:
	$(info CARLsim $(SIM_MAJOR_NUM).$(SIM_MINOR_NUM).$(SIM_BUILD_NUM) successfully installed.)

farewell:
	$(info CARLsim $(SIM_MAJOR_NUM).$(SIM_MINOR_NUM).$(SIM_BUILD_NUM) successfully uninstalled.)
