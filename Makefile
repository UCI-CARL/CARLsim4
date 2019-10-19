##----------------------------------------------------------------------------##
##
##   CARLsim4 Main Makefile
##   ----------------------
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

# the following are filled in the include files and passed up
objects_cpp :=
objects_cu :=
output :=
output_folders := doc/html

.PHONY: default clean distclean help
default: release

include carlsim/configure.mk   # import configuration settings
include carlsim/carlsim.mk     # import CARLsim-related variables and rules
include carlsim/libcarlsim.mk  # import libCARLsim-related variables and rules
include carlsim/test.mk        # import test-related variables and rules

# clean all objects
clean:
	$(RM) $(objects_cpp) $(objects_cu)

# clean all objects and output files
distclean:
	$(RM) $(objects_cpp) $(objects_cu) $(output)
	$(RMR) $(output_folders)

# print a help message
help:
	@ echo 
	@ echo "CARLsim4 Makefile options:"
	@ echo 
	@ echo "make                Compiles CARLsim4 in default mode (release)"
	@ echo "make release        Compiles CARLsim4 in release mode (-O3)"
	@ echo "make release_nocuda Compiles CARLsim4 in release mode without CUDA library"
	@ echo "make debug          Compiles CARLsim4 in debug mode (-O0 -g -Wall)"
	@ echo "make debug_nocuda   Compiles CARLsim4 in debug mode without CUDA library"
	@ echo "make -E install     Installs CARLsim4 library (make sure -E is set; may"
	@ echo "                    require root privileges)"
	@ echo "make -E uninstall   Uninstalls CARLsim4 library (make sure -E is set; may"
	@ echo "                    require root privileges)"
	@ echo "make clean          Cleans out all object files"
	@ echo "make distclean      Cleans out all object and output files"
	@ echo "make help           Brings up this message"
