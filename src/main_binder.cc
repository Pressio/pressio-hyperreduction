
#ifndef PRESSIOTOOLS_MAIN_BINDER_HPP_
#define PRESSIOTOOLS_MAIN_BINDER_HPP_

#include "types.hpp"
#include "./data_structures/multivector.hpp"
#include "./data_structures/vector.hpp"
#include "tsqr.hpp"
#include "svd.hpp"
#include "pinv.hpp"

#define PT_STRINGIFY(x) #x
#define PT_MACRO_STRINGIFY(x) PT_STRINGIFY(x)

PYBIND11_MODULE(pressiotools, mParent)
{
  mParent.attr("__version__") = PT_MACRO_STRINGIFY(VERSION_IN);

  // bind multivector
  using mv_t = pressiotools::MultiVector;
  pybind11::class_<mv_t> mv(mParent, "MultiVector");
  mv.def(pybind11::init<pressiotools::py_f_arr>());
  mv.def(pybind11::init<pressiotools::py_c_arr>());
  mv.def("data", &mv_t::data);
  mv.def("extentLocal", &mv_t::extentLocal);
  mv.def("extentGlobal", &mv_t::extentGlobal);

  // bind vector
  using v_t = pressiotools::Vector;
  pybind11::class_<v_t> vec(mParent, "Vector");
  vec.def(pybind11::init<pressiotools::py_f_arr>());
  vec.def(pybind11::init<pressiotools::py_c_arr>());
  vec.def("data", &v_t::data);
  vec.def("extentLocal", &v_t::extentLocal);
  vec.def("extentGlobal", &v_t::extentGlobal);

  // bind tsqr
  using tsqr_t = pressiotools::Tsqr;
  pybind11::class_<tsqr_t> tsqr(mParent, "Tsqr");
  tsqr.def(pybind11::init());
  tsqr.def("computeThinOutOfPlace", &tsqr_t::computeThinOutOfPlace);
  tsqr.def("viewR", &tsqr_t::viewR, pybind11::return_value_policy::reference);
  tsqr.def("viewLocalQ", &tsqr_t::viewLocalQ, pybind11::return_value_policy::reference);

  // bind svd
  using svd_t = pressiotools::Svd;
  pybind11::class_<svd_t> svd(mParent, "svd");
  svd.def(pybind11::init());
  svd.def("computeThin", &svd_t::computeThin);
  svd.def("viewSingValues",
	  &svd_t::viewS, pybind11::return_value_policy::reference);
  svd.def("viewLocalLeftSingVectors",
	  &svd_t::viewU, pybind11::return_value_policy::reference);
  svd.def("viewRightSingVectorsT",
	  &svd_t::viewVT, pybind11::return_value_policy::reference);

  // bind pinv
  using pinv_t = pressiotools::Pinv;
  pybind11::class_<pinv_t> pinv(mParent, "pinv");
  pinv.def(pybind11::init());
  pinv.def("compute", &pinv_t::compute);
  pinv.def("viewLocalAstarT",
	   &pinv_t::viewLocalAstarT, pybind11::return_value_policy::reference);
  pinv.def("apply",
	   &pinv_t::apply, pybind11::return_value_policy::reference);
};
#endif
