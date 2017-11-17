# Targets

    add_library(carlsim-monitor
        connection_monitor_core.cpp
        connection_monitor.cpp
        group_monitor_core.cpp
        group_monitor.cpp
        spike_monitor_core.cpp
        spike_monitor.cpp
    )

# Properties

    # Since we build shared library enable position independent code
    set_property(TARGET carlsim-monitor PROPERTY
        POSITION_INDEPENDENT_CODE TRUE)

# Includes

    target_include_directories(carlsim-monitor
        PUBLIC
            .
    )

# Linking

    target_link_libraries(carlsim-monitor
        PRIVATE
            carlsim-kernel
    )

# Installation

    install(
        FILES
            connection_monitor_core.h
            connection_monitor.h
            group_monitor_core.h
            group_monitor.h
            spike_monitor_core.h
            spike_monitor.h
        DESTINATION include)
