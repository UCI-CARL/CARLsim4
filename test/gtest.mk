# GTEST

# Points to the root of Google Test, relative to where this file is.
# Remember to tweak this if you move this file.
GTEST_DIR = /opt/gtest

# Our local, project-specific compilation of Google Test
GTEST_LIB_DIR = test/lib

# Where to find user code.
USER_DIR = $(GTEST_DIR)/samples


# GTEST

# Flags passed to the preprocessor.
# Set Google Test's header directory as a system directory, such that
# the compiler doesn't generate warnings in Google Test headers.
GTEST_CPPFLAGS += -isystem $(GTEST_DIR)/include -DGTEST_LINKED_AS_SHARED_LIBRARY=1

# Flags passed to the C++ compiler.
GTEST_CXXFLAGS += -g -Wall -Wextra -pthread


# All Google Test headers.  Usually you shouldn't change this
# definition.
GTEST_HEADERS = $(GTEST_DIR)/include/gtest/*.h \
                $(GTEST_DIR)/include/gtest/internal/*.h



# For simplicity and to avoid depending on Google Test's
# implementation details, the dependencies specified below are
# conservative and not optimized.  This is fine as Google Test
# compiles fast and for ordinary users its source rarely changes.
gtest: $(GTEST_LIB_DIR)/libgtest.a $(GTEST_LIB_DIR)/libgtest_main.a

$(GTEST_LIB_DIR)/libgtest.a : $(GTEST_LIB_DIR)/gtest-all.o
	$(AR) $(ARFLAGS) $@ $^

$(GTEST_LIB_DIR)/libgtest_main.a : $(GTEST_LIB_DIR)/gtest-all.o $(GTEST_LIB_DIR)/gtest_main.o
	$(AR) $(ARFLAGS) $@ $^

$(GTEST_LIB_DIR)/gtest-all.o : $(GTEST_SRCS_)
	mkdir -p $(GTEST_LIB_DIR)
	$(CXX) $(GTEST_CPPFLAGS) -I$(GTEST_DIR) $(GTEST_CXXFLAGS) -c \
            $(GTEST_DIR)/src/gtest-all.cc -o $@ $^

$(GTEST_LIB_DIR)/gtest_main.o : $(GTEST_SRCS_)
	mkdir -p $(GTEST_LIB_DIR)
	$(CXX) $(GTEST_CPPFLAGS) -I$(GTEST_DIR) $(GTEST_CXXFLAGS) -c \
            $(GTEST_DIR)/src/gtest_main.cc -o $@ $^
