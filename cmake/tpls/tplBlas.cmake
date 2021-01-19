
option(PRESSIO_TOOLS_ENABLE_TPL_BLAS "Enable Blas TPL" OFF)

if(PRESSIO_TOOLS_ENABLE_TPL_BLAS OR PRESSIO_TOOLS_ENABLE_TPL_TRILINOS)

if (BLAS_INCLUDE_DIRS AND BLAS_LIBRARY_DIRS AND BLAS_LIBRARIES)
message("")
include_directories(${BLAS_INCLUDE_DIRS})
link_directories($BLAS_LIBRARY_DIRS})
link_libraries(${BLAS_LIBRARIES})

else()

find_package( BLAS REQUIRED )
link_libraries(${BLAS_LIBRARIES})
message("")

endif()
endif()
