# Targets

    add_library(carlsim-visual-stimulus STATIC
        visual_stimulus.cpp
    )

# Linking

    target_link_libraries(carlsim-visual-stimulus
        PRIVATE
            carlsim-interface
    )
