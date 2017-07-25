# Targets

    add_library(carlsim-kernel
        src/print_snn_info.cpp
        src/snn_cpu_module.cpp
        src/snn_manager.cpp
        src/spike_buffer.cpp
    )

# Linking

    if(UNIX)
        target_link_libraries(carlsim-kernel
            PRIVATE
                pthread
        )
    endif()
