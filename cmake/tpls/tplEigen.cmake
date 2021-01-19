
option(PRESSIO_TOOLS_ENABLE_TPL_EIGEN "Enable Eigen TPL" ON)

if(PRESSIO_TOOLS_ENABLE_TPL_EIGEN)
  message("Found PRESSIO_TOOLS_ENABLE_TPL_EIGEN=${PRESSIO_TOOLS_ENABLE_TPL_EIGEN}.")

  if(NOT EIGEN_INC_DIR AND NOT EIGEN_INCLUDE_DIR)
    message(FATAL_ERROR
      "I cannot find the Eigen headers. Please reconfigure with:
      	-DEIGEN_INC_DIR=<full-path-to-headers>
      	or
      	-DEIGEN_INCLUDE_DIR=<full-path-to-headers>
      ")
  endif()

  if(NOT EIGEN_INC_DIR AND EIGEN_INCLUDE_DIR)
    set(EIGEN_INC_DIR ${EIGEN_INCLUDE_DIR})
  endif()

  include_directories(${EIGEN_INC_DIR})

endif()
