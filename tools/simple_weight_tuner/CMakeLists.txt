# Targets

    add_library(carlsim-simple-weight-tuner STATIC
        simple_weight_tuner.cpp
    )

# Properties

    # Since we build shared library enable position independent code
    set_property(TARGET carlsim-simple-weight-tuner PROPERTY
        POSITION_INDEPENDENT_CODE TRUE)

# Linking

    target_link_libraries(carlsim-simple-weight-tuner
        PRIVATE
            carlsim-interface
            carlsim-monitor
    )

# Installation

    install(FILES simple_weight_tuner.h DESTINATION include)
