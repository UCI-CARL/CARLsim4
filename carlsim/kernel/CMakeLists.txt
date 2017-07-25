# Targets

    add_library(carlsim-kernel
        src/print_snn_info.cpp
        src/snn_cpu_module.cpp
        src/snn_manager.cpp
        src/spike_buffer.cpp
    )

# Definitions

    if(CARLSIM_NO_CUDA)
        target_compile_definitions(carlsim-kernel
            PUBLIC
                -D__NO_CUDA__
        )
    endif()

    target_include_directories(carlsim-kernel
        PUBLIC
            inc
    )

# Linking

    if(UNIX)
        target_link_libraries(carlsim-kernel
            PRIVATE
                pthread
        )
    endif()
