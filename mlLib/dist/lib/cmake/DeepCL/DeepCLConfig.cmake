# - Config file for the DeepCL package
# It defines the following variables
#  DeepCL_INCLUDE_DIRS - include directories for DeepCL
#  DeepCL_LIBRARIES    - libraries to link against
 
get_filename_component(DEEPCL_CMAKE_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
set(DeepCL_INCLUDE_DIRS "${DEEPCL_CMAKE_DIR}/../../../include;${DEEPCL_CMAKE_DIR}/../../../include/deepcl;${DEEPCL_CMAKE_DIR}/../../../include/easycl")
 
if(NOT TARGET DeepCL AND NOT DeepCL_BINARY_DIR)
  include("${DEEPCL_CMAKE_DIR}/DeepCLTargets.cmake")
endif()
   
set(DeepCL_LIBRARIES "DeepCL;EasyCL")
