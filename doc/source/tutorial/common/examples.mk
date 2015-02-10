# module include file for common files used by our examples
# edited by KDC

local_dir := examples/common
local_deps := stimGenerator.h writeSpikeToArray.h
local_src := stimGenerator.cpp stimGenerator.h writeSpikeToArray.h
sources += $(local_src)
common_objs += $(addprefix $(local_dir)/,stimGenerator.o)
common_sources += $(local_src)
objects += $(common_objs)

# rules to build the common objects
$(local_dir)/%.o: %.cpp $(local_deps)
	$(CC) -c $(CPPFLAGS) $< -o $@
