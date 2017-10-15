project(carlsim)
cmake_minimum_required(VERSION 3.0)

# Options

    option(CARLSIM_NO_CUDA "Turn off CUDA support" OFF)

# Targets

    add_library(carlsim SHARED carlsim/dummy.cpp)

# Properties

    set_property(TARGET carlsim PROPERTY VERSION "4.0.0")

# Linking

    # Combine archives into shared library using this macro
    macro(link_archive name)
        target_link_libraries(carlsim PUBLIC
            -Wl,--whole-archive ${name} -Wl,--no-whole-archive
        )
    endmacro()

    link_archive(carlsim-interface)
    link_archive(carlsim-kernel)
    link_archive(carlsim-monitor)
    link_archive(carlsim-simple-weight-tuner)
    link_archive(carlsim-spike-generators)
    link_archive(carlsim-stopwatch)
    link_archive(carlsim-visual-stimulus)

# Installation

    install(TARGETS carlsim DESTINATION lib)
    install(FILES cmake/carlsim-config.cmake DESTINATION lib/cmake/carlsim)

# Subdirectories

    add_subdirectory(carlsim)
    add_subdirectory(tools)
