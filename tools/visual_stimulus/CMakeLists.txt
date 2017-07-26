# Targets

    add_library(carlsim-visual-stimulus STATIC
        visual_stimulus.cpp
    )

# Properties

    # Since we build shared library enable position independent code
    set_property(TARGET carlsim-visual-stimulus PROPERTY
        POSITION_INDEPENDENT_CODE TRUE)

# Linking

    target_link_libraries(carlsim-visual-stimulus
        PRIVATE
            carlsim-interface
    )

# Installation

    install(FILES visual_stimulus.h DESTINATION include)
