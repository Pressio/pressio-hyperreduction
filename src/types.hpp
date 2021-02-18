
#ifndef PRESSIOTOOLS_TYPES_HPP_
#define PRESSIOTOOLS_TYPES_HPP_

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#ifdef PRESSIOTOOLS_ENABLE_TPL_MPI
#include "mpi.h"
#endif

namespace pressiotools
{

using scalar_t	= double;
using py_c_arr	= pybind11::array_t<scalar_t, pybind11::array::c_style>;
using py_f_arr	= pybind11::array_t<scalar_t, pybind11::array::f_style>;

}
#endif
