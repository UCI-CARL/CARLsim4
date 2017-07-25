# Targets

    add_library(carlsim-monitor
        connection_monitor_core.cpp
        connection_monitor.cpp
        group_monitor_core.cpp
        group_monitor.cpp
        spike_monitor_core.cpp
        spike_monitor.cpp
    )

# Includes

    target_include_directories(carlsim-monitor
        PUBLIC
            .
    )
