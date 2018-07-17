find_library(CARLSIM_LIBRARY carlsim)
find_path(CARLSIM_INCLUDE_DIR carlsim.h)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(carlsim DEFAULT_MSG
    CARLSIM_LIBRARY
    CARLSIM_INCLUDE_DIR
)

add_library(carlsim SHARED IMPORTED)
set_target_properties(carlsim PROPERTIES
    IMPORTED_LOCATION ${CARLSIM_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES ${CARLSIM_INCLUDE_DIR}
)

mark_as_advanced(
    CARLSIM_LIBRARY
    CARLSIM_INCLUDE_DIR
)
