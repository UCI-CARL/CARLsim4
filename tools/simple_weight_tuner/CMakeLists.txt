# Targets

    add_library(carlsim-simple-weight-tuner STATIC
        simple_weight_tuner.cpp
    )

# Linking

    target_link_libraries(carlsim-simple-weight-tuner
        PRIVATE
            carlsim-interface
            carlsim-monitor
    )
