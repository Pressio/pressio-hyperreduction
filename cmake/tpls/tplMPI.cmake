
option(PRESSIO_TOOLS_ENABLE_TPL_MPI "Enable MPI" ON)
if(PRESSIO_TOOLS_ENABLE_TPL_MPI)
  find_package(MPI REQUIRED)
  include_directories(${MPI_CXX_INCLUDE_PATH} ${MPI_C_INCLUDE_PATH})

  if (${MPIEXEC_EXECUTABLE} STREQUAL "MPIEXEC_EXECUTABLE-NOTFOUND")
    message(FATAL_ERROR "${MPIEXEC_EXECUTABLE} not found, check that MPI is in your \
path or set this variable explicitly at configure time")
  endif()
endif()
