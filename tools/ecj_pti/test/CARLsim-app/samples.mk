# Builds a sample test.  A test should link with either gtest.a or
# gtest_main.a, depending on whether it defines its own main()
# function. Tests that define their own main() function should link to
# gtest.a while those that do not should link to gtest_main.a.
sample1.o : $(USER_DIR)/sample1.cc $(USER_DIR)/sample1.h 
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(USER_DIR)/sample1.cc

sample1_unittest.o : $(USER_DIR)/sample1_unittest.cc \
                     $(USER_DIR)/sample1.h 
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(USER_DIR)/sample1_unittest.cc

sample1_unittest : sample1.o sample1_unittest.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -lpthread -L$(GTEST_LIB_DIR) -lgtest_main $^ -o $@

sample2.o : $(USER_DIR)/sample2.cc $(USER_DIR)/sample2.h 
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(USER_DIR)/sample2.cc

sample2_unittest.o : $(USER_DIR)/sample2_unittest.cc \
                     $(USER_DIR)/sample2.h 
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(USER_DIR)/sample2_unittest.cc

sample2_unittest : sample2.o sample2_unittest.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -lpthread -L$(GTEST_LIB_DIR) -lgtest_main $^ -o $@

sample3_unittest.o : $(USER_DIR)/sample3_unittest.cc \
                     $(USER_DIR)/sample3-inl.h 
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(USER_DIR)/sample3_unittest.cc

sample3_unittest : sample3_unittest.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -lpthread -L$(GTEST_LIB_DIR) -lgtest_main $^ -o $@

sample4.o : $(USER_DIR)/sample4.cc $(USER_DIR)/sample4.h 
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(USER_DIR)/sample4.cc

sample4_unittest.o : $(USER_DIR)/sample4_unittest.cc \
                     $(USER_DIR)/sample4.h 
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(USER_DIR)/sample4_unittest.cc

sample4_unittest : sample4.o sample4_unittest.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -lpthread -L$(GTEST_LIB_DIR) -lgtest_main $^ -o $@

sample5_unittest.o : $(USER_DIR)/sample5_unittest.cc \
	$(USER_DIR)/sample1.h $(USER_DIR)/sample3-inl.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(USER_DIR)/sample5_unittest.cc

sample5_unittest : sample5_unittest.o sample1.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -lpthread -L$(GTEST_LIB_DIR) -lgtest_main $^ -o $@

sample6_unittest.o : $(USER_DIR)/sample6_unittest.cc \
	$(USER_DIR)/prime_tables.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(USER_DIR)/sample6_unittest.cc

sample6_unittest : sample6_unittest.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -lpthread -L$(GTEST_LIB_DIR) -lgtest_main $^ -o $@

sample7_unittest.o : $(USER_DIR)/sample7_unittest.cc \
	$(USER_DIR)/prime_tables.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(USER_DIR)/sample7_unittest.cc

sample7_unittest : sample7_unittest.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -lpthread -L$(GTEST_LIB_DIR) -lgtest_main $^ -o $@

sample8_unittest.o : $(USER_DIR)/sample8_unittest.cc \
	$(USER_DIR)/prime_tables.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(USER_DIR)/sample8_unittest.cc

sample8_unittest : sample8_unittest.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -lpthread -L$(GTEST_LIB_DIR) -lgtest_main $^ -o $@

sample9_unittest.o : $(USER_DIR)/sample9_unittest.cc \
	$(USER_DIR)/prime_tables.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(USER_DIR)/sample9_unittest.cc

sample9_unittest : sample9_unittest.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -lpthread -L$(GTEST_LIB_DIR) $^ -o $@ -lgtest

sample10_unittest.o : $(USER_DIR)/sample10_unittest.cc \
	$(USER_DIR)/prime_tables.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(USER_DIR)/sample10_unittest.cc

sample10_unittest : sample10_unittest.o
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -lpthread -L$(GTEST_LIB_DIR) $^ -o $@ -lgtest
