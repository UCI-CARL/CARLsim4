project(carlsim)
cmake_minimum_required(VERSION 3.0)

# Options

    option(CARLSIM_NO_CUDA "Turn off CUDA support" OFF)

# Global definitions

    if(CARLSIM_NO_CUDA)
        add_definitions(-D__NO_CUDA__)
    endif()

# Subdirectories

    add_subdirectory(carlsim)
