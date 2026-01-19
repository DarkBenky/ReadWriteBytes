#----------------------------------------------------------------
# Generated CMake target import file for configuration "RelWithDebInfo".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "EasyCL" for configuration "RelWithDebInfo"
set_property(TARGET EasyCL APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(EasyCL PROPERTIES
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELWITHDEBINFO "clew;dl"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/lib/libEasyCL.so"
  IMPORTED_SONAME_RELWITHDEBINFO "libEasyCL.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS EasyCL )
list(APPEND _IMPORT_CHECK_FILES_FOR_EasyCL "${_IMPORT_PREFIX}/lib/libEasyCL.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
