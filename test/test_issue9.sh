# script to test for memory leaks
make -f Makefile.test test_issue9
valgrind --tool=memcheck --leak-check=yes --show-reachable=yes --num-callers=20 --track-fds=yes ./test_issue9

# TODO: automatically detect whether network passed or failed the valgrind test
