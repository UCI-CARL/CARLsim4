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

# Linking

    target_link_libraries(carlsim-tests
        PRIVATE
            carlsim-spike-generators
            gtest
    )
