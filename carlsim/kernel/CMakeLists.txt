# Dependencies

    if(NOT CARLSIM_NO_CUDA)
        find_package(CUDA REQUIRED)
    endif()

# Targets

    if(NOT CARLSIM_NO_CUDA)
        cuda_add_library(carlsim-kernel-cuda
            src/gpu_module/snn_gpu_module.cu
            OPTIONS
                -Xcompiler -fPIC
                -gencode arch=compute_30,code=sm_30
        )
    endif()

    add_library(carlsim-kernel
        src/print_snn_info.cpp
        src/snn_cpu_module.cpp
        src/snn_manager.cpp
        src/spike_buffer.cpp
    )

# Properties

    # Since we build shared library enable position independent code
    set_property(TARGET carlsim-kernel PROPERTY
        POSITION_INDEPENDENT_CODE TRUE)

# Definitions

    if(CARLSIM_NO_CUDA)
        target_compile_definitions(carlsim-kernel
            PUBLIC
                -D__NO_CUDA__
        )
    else()
        target_compile_definitions(carlsim-kernel-cuda
            PUBLIC
                -D__CUDA${CUDA_VERSION_MAJOR}__
        )
    endif()

# Includes

    if(NOT CARLSIM_NO_CUDA)
        target_include_directories(carlsim-kernel-cuda
            PUBLIC
                inc
                ${CUDA_INCLUDE_DIRS}
                ${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc
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

    if(NOT CARLSIM_NO_CUDA)
        target_link_libraries(carlsim-kernel-cuda
            ${CUDA_curand_LIBRARY}
        )

        target_link_libraries(carlsim-kernel
            PUBLIC
                carlsim-kernel-cuda
        )
    endif()

# Installation

    install(
        FILES
            inc/cuda_version_control.h
            inc/error_code.h
            inc/snn_datastructures.h
            inc/snn_definitions.h
            inc/snn.h
            inc/spike_buffer.h
        DESTINATION include)
