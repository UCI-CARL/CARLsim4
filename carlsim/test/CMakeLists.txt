# Dependencies

    find_package(GTest REQUIRED)

# Targets

    add_executable(carlsim-tests
        carlsim_tests_common.cpp
        coba.cpp
        connect.cpp
        conn_mon.cpp
        core.cpp
        cuba.cpp
        group_mon.cpp
        interface.cpp
        main.cpp
        multi_runtimes.cpp
        poiss_rate.cpp
        spike_gen.cpp
        spike_mon.cpp
        stdp.cpp
        stp.cpp
    )

# Includes

    target_include_directories(carlsim-tests
        PRIVATE
            ${GTEST_INCLUDE_DIRS}
    )

# Linking

    target_link_libraries(carlsim-tests
        PRIVATE
            carlsim-spike-generators
            ${GTEST_LIBRARIES}
    )
