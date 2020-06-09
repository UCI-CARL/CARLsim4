##----------------------------------------------------------------------------##
##
##   CARLsim4 CARLsim-ECJ PTI interface Makefile
##   -------------------------
##
##   Authors:   Eric Scott <escott8@gmu.edu>
##              Kristofor Carlson <kdcarlso@uci.edu>
##		Michael Beyeler <mbeyeler@uci.edu>
##
##   Institute: Cognitive Anteater Robotics Lab (CARL)
##              Department of Cognitive Sciences
##              University of California, Irvine
##              Irvine, CA, 92697-5100, USA
##
##   Version:   04/11/2018
##
##----------------------------------------------------------------------------##

################################################################################
# Start of user-modifiable section
################################################################################

# Directory where all include files reside. The Makefile will automatically
# detect and include all .h files within that directory.
proj_inc_dir   := inc

# Directory where all source files reside. The Makefile will automatically
# detect and include all .cpp and .cu files within that directory.
proj_src_dir   := src

# Name of the library file to be created.
lib_pti_name := libCARLsimPTI.a
jar_target := $(proj_src_dir)/java/dist/CARLsim-ECJ.jar


################################################################################
# End of user-modifiable section
################################################################################


#------------------------------------------------------------------------------
# Include configuration file
#------------------------------------------------------------------------------

# NOTE: If your CARLsim4 installation does not reside in the default path, make
# sure the environment variable CARLSIM4_INSTALL_DIR is set.
ifneq ($(CARLSIM4_INSTALL_DIR),)
	CARLSIM4_INC_DIR  := $(CARLSIM4_INSTALL_DIR)/include
else
	CARLSIM4_INC_DIR  := /usr/local/include/carlsim
endif

# include compile flags etc.
include $(CARLSIM4_INC_DIR)/configure.mk

# Version info
num_ver := $(SIM_MAJOR_NUM).$(SIM_MINOR_NUM)
lib_ver := $(num_ver).$(SIM_BUILD_NUM)
lib_target := $(lib_pti_name).$(lib_ver)

#------------------------------------------------------------------------------
# Build local variables
#------------------------------------------------------------------------------

# build list of all .cpp, .cu, and .h files (but don't include main_src_file)
cpp_files  := $(wildcard $(proj_src_dir)/*.cpp)
cu_files   := $(wildcard $(proj_src_dir)/*.cu)
inc_files  := $(wildcard $(proj_inc_dir)/**/*.h)

obj_files    := $(patsubst $(proj_src_dir)/%.cpp, obj/%.o, $(cpp_files))

# handled by clean
clean_files := $(obj_files) $(lib_pti_name).$(lib_ver) $(wildcard $(proj_src_dir)/java/dist) $(wildcard $(proj_src_dir)/java/build)

#------------------------------------------------------------------------------
# Project targets and rules
#------------------------------------------------------------------------------

.PHONY: clean distclean help
default: $(lib_target) $(jar_target)


$(lib_target): $(cpp_files) $(obj_files) $(inc_files)
	ar rcs $@ $(obj_files)

obj/%.o: $(proj_src_dir)/%.cpp $(inc_files)
	@test -d obj/ || mkdir -p obj/
	$(CXX) -c $(CXXINCFL) $(CXXFL) -I${proj_inc_dir} $< -o $@

# Build the CARLsim-ECJ.jar from nbproject using ant
$(jar_target):
	cd $(proj_src_dir)/java; ant jar -Dendorsed.classpath=$(ECJ_JAR)

install_jar: $(jar_target)
	@test -d $(ECJ_PTI_DIR) || mkdir -p $(ECJ_PTI_DIR)
	@test -d $(ECJ_PTI_DIR)/lib || mkdir -p $(ECJ_PTI_DIR)/lib
	@install -m 0755 $(jar_target) $(ECJ_PTI_DIR)/lib

install_carlsim_pti: $(lib_target)
	@test -d $(ECJ_PTI_DIR) || mkdir -p $(ECJ_PTI_DIR)
	@test -d $(ECJ_PTI_DIR)/lib || mkdir $(ECJ_PTI_DIR)/lib
	@test -d $(ECJ_PTI_DIR)/inc || mkdir $(ECJ_PTI_DIR)/inc
	@install -m 0755 $(lib_target) $(ECJ_PTI_DIR)/lib
	@ln -fs $(ECJ_PTI_DIR)/lib/$(lib_pti_name).$(lib_ver) \
		$(ECJ_PTI_DIR)/lib/$(lib_pti_name).$(num_ver)
	@ln -fs $(ECJ_PTI_DIR)/lib/$(lib_pti_name).$(num_ver) \
		$(ECJ_PTI_DIR)/lib/$(lib_pti_name)
	@install -m 0644 $(proj_inc_dir)/*.h \
		$(ECJ_PTI_DIR)/inc

install: install_carlsim_pti install_jar

clean:
	$(RM) -r $(clean_files)

help:
	$(info CARLsim-ECJ Makefile options:)
	$(info )
	$(info make	       Compiles $(lib_pti_name) and CARLsim-ECJ.jar)
	$(info make clean	 Cleans out all compiled files)
	$(info make help	  Brings up this message)
