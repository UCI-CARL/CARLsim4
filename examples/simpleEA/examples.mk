# module include file for simpleEA example program
example := simpleEA
output := *.dot *.txt *.log tmp* *.status

# local info (vars can be overwritten)
local_dir := examples/$(example)
local_src := $(local_dir)/main_$(example).cpp
local_prog := $(local_dir)/$(example)

# info passed up to Makefile
sources += $(local_src)
output_files += $(addprefix $(local_dir)/,$(output))
pti_programs += $(local_prog)

# Rules for example binaries that use EO/PTI
.PHONY: simpleEA
simpleEA: $(local_src) $(local_prog)

$(local_prog): $(local_src)
	$(CC) $(CPPFLAGS) $(EO_FLAGS) $(PTI_FLAGS) $< -o $@ $(PTI_LIBS) $(EO_LIBS)
