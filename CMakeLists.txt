project(carlsim)
cmake_minimum_required(VERSION 3.0)

# Options

    option(CARLSIM_NO_CUDA "Turn off CUDA support" OFF)

# Subdirectories

    add_subdirectory(carlsim)
    add_subdirectory(tools)
