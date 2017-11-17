# Targets

    add_library(carlsim-spike-generators
        interactive_spikegen.cpp
        periodic_spikegen.cpp
        pre_post_group_spikegen.cpp
        spikegen_from_file.cpp
        spikegen_from_vector.cpp
    )

# Properties

    # Since we build shared library enable position independent code
    set_property(TARGET carlsim-spike-generators PROPERTY
        POSITION_INDEPENDENT_CODE TRUE)

# Includes

    target_include_directories(carlsim-spike-generators
        PUBLIC
            .
    )

# Linking

    target_link_libraries(carlsim-spike-generators
        PUBLIC
            carlsim-interface
            carlsim-monitor
    )

# Installation

    install(
        FILES
            interactive_spikegen.h
            periodic_spikegen.h
            pre_post_group_spikegen.h
            spikegen_from_file.h
            spikegen_from_vector.h
        DESTINATION include)
