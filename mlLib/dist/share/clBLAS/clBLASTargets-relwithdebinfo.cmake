#----------------------------------------------------------------
# Generated CMake target import file for configuration "RelWithDebInfo".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "clBLAS" for configuration "RelWithDebInfo"
set_property(TARGET clBLAS APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(clBLAS PROPERTIES
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELWITHDEBINFO "/home/ubuntu/jenkins/workspace/deepcl-linux64-cpp-2/dist/lib/libclew.so;m"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/lib/libclBLAS.so.2.4.0"
  IMPORTED_SONAME_RELWITHDEBINFO "libclBLAS.so.2"
  )

list(APPEND _IMPORT_CHECK_TARGETS clBLAS )
list(APPEND _IMPORT_CHECK_FILES_FOR_clBLAS "${_IMPORT_PREFIX}/lib/libclBLAS.so.2.4.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
