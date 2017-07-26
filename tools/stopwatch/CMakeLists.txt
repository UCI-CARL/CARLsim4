# Targets

    add_library(carlsim-stopwatch STATIC
        stopwatch.cpp
    )

# Linking

    target_link_libraries(carlsim-stopwatch
        PRIVATE
            carlsim-interface
    )
