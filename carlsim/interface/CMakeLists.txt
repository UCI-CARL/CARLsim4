# Targets

    add_library(carlsim-interface
        src/callback_core.cpp
        src/carlsim.cpp
        src/linear_algebra.cpp
        src/poisson_rate.cpp
        src/user_errors.cpp
    )

# Includes

    target_include_directories(carlsim-interface
        PUBLIC
            inc
    )

# Linking

    target_link_libraries(carlsim-interface
        PUBLIC
            carlsim-kernel
    )
