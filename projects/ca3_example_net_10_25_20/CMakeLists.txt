project(carlsim-hello-world)
cmake_minimum_required(VERSION 3.0)

# Dependencies

    find_package(carlsim REQUIRED)

# Targets

    add_executable(hello_world
        src/main_hello_world.cpp
    )

# Linking

    target_link_libraries(hello_world
        PRIVATE
            carlsim
    )
