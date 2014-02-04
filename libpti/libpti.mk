# module include file for pti library
# edited by KDC

#-------------------------------------------------------------------------------
# Release information
#-------------------------------------------------------------------------------
pti_major_ver := 1
pti_minor_ver := 0
pti_rel_ver   := 0

#-------------------------------------------------------------------------------
# libpti flags
#-------------------------------------------------------------------------------
LIB_FLAGS = -fPIC
DL_FLAGS = -shared
PTI_FLAGS = -I$(PTI_INSTALL_DIR)/include -L$(PTI_INSTALL_DIR)/lib
PTI_LIBS = -lpti_make_algo_scalar_es \
         -lmake_checkpoint_real \
         -lmake_op_real \
         -lpti_make_algo_scalar_real \
         -lmake_continue_real

EO_FLAGS = -I$(EO_INSTALL_DIR)/src -L$(EO_INSTALL_DIR)/release/lib
EO_LIBS = -leo -lga -leoutils -lcma -les

local_dir  := libpti
local_lib  := $(addprefix $(local_dir)/,libpti_make_algo_scalar_es.so \
	 libpti_make_algo_scalar_real.so \
	 libmake_op_real.so \
	 libmake_continue_real.so \
	 libmake_checkpoint_real.so)
local_src  := $(addprefix $(local_dir)/,pti_make_algo_scalar_real.cpp \
           pti_make_algo_scalar_es.cpp)
local_objs := $(subst .cpp,.o,$(local_src))
local_deps := pti_eoAlgo.h pti_eoEasyEA.h \
	pti_eoCombinedContinue.h \
	pti_eoGenContinue.h \
	pti_make_algo_scalar.h \
	pti_make_es.h \
        pti_make_real.h pti.h

libraries += $(wildcard $(local_dir)/*.so*)
sources += $(local_src)

#-------------------------------------------------------------------------------
# Rules to build the pti library
#-------------------------------------------------------------------------------
# Rules to build library pti object files
.PHONY: pti
pti: $(local_deps) $(local_src) $(local_lib)

$(local_dir)/%.o: %.cpp
	$(CC) -c $(CPPFLAGS) -I$(CURDIR)/$(inc_dir) $(EO_FLAGS) \
	$(LIB_FLAGS) $< -o $@

$(local_dir)/lib%.so: $(local_dir)/%.o
	$(CC) $(CPPFLAGS) $(DL_FLAGS) -o \
	$@.$(pti_major_ver).$(pti_minor_ver).$(pti_rel_ver) $<; \
	ln -fs lib$*.so.$(pti_major_ver).$(pti_minor_ver).$(pti_rel_ver) \
	$@.$(pti_major_ver).$(pti_minor_ver); \
	ln -fs lib$*.so.$(pti_major_ver).$(pti_minor_ver) $@

# maybe move this to main makefile.  It should make sure the other things are built first (deps?)
.PHONY: install
install: $(local_deps) $(local_src) $(local_lib)
	-mkdir -p $(PTI_INSTALL_DIR)
	-mkdir -p $(PTI_INSTALL_DIR)/lib
	-mkdir -p $(PTI_INSTALL_DIR)/include
	cp -R $(lib_dir)/*.so* $(PTI_INSTALL_DIR)/lib
	cp -R $(inc_dir)/*.h $(PTI_INSTALL_DIR)/include

.PHONY: uninstall
uninstall: 
	$(RM) $(PTI_INSTALL_DIR)
