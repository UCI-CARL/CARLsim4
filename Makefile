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
##   Version:   03/30/2016
##
##----------------------------------------------------------------------------##
	
# the following are filled in the include files and passed up
targets :=
objects :=
libraries :=
output_folders := doc/html

.PHONY: default clean distclean release debug
default: release

include carlsim/configure.mk   # import configuration settings
include carlsim/carlsim.mk     # import CARLsim-related variables and rules
include carlsim/libcarlsim.mk  # import libCARLsim-related variables and rules

# clean all objects
clean:
	$(RM) $(objects)

# clean all objects and output files
distclean:
	$(RM) $(objects) $(targets) $(libraries)
	$(RMR) $(output_folders)

# print a help message
help:
	@ echo 
	@ echo "CARLsim4 Makefile options:"
	@ echo 
	@ echo "make            Compiles CARLsim4 in default mode (release)"
	@ echo "make release    Compiles CARLsim4 in release mode (no debug output,"
	@ echo "                using fast math and GPU optimization level 3)"
	@ echo "make debug      Compiles CARLsim4 in debug mode (-g -Wall)"
	@ echo "make install    Installs CARLsim4 library (may require root privileges)"
	@ echo "make clean      Cleans out all object files"
	@ echo "make distclean  Cleans out all object and output files"
	@ echo "make help       Brings up this message"