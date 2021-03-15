
#ifndef PRESSIOTOOLS_MAIN_BINDER_HPP_
#define PRESSIOTOOLS_MAIN_BINDER_HPP_

#include <iostream>
#include "types.hpp"
#include "./data_structures/vector.hpp"
#include "./data_structures/multivector.hpp"

#ifdef PRESSIOTOOLS_ENABLE_TPL_TRILINOS
#include "ops.hpp"
#include "tsqr.hpp"
#include "svd.hpp"
#include "pinv.hpp"
#endif

PYBIND11_MODULE(MODNAME, mParent)
{
  // bind vector
  using v_t = pressiotools::Vector;
  pybind11::class_<v_t> vec(mParent, "Vector");
  vec.def(pybind11::init<pressiotools::py_f_arr>());
  vec.def(pybind11::init<pressiotools::py_c_arr>());
  vec.def(pybind11::init<typename v_t::ord_t>());
  vec.def("data",	  &v_t::data);
  vec.def("address",	  &v_t::address);
  vec.def("extentLocal",  &v_t::extentLocal);
  vec.def("extentGlobal", &v_t::extentGlobal);
  vec.def("sumGlobal",	  &v_t::sumGlobal);
  vec.def("sumLocal",     &v_t::sumLocal);
#ifndef PRESSIOTOOLS_ENABLE_TPL_MPI
  vec.def("extent",	  &v_t::extent);
  vec.def("sum",	  &v_t::sum);
#endif

  // bind multivector
  using mv_t = pressiotools::MultiVector;
  pybind11::class_<mv_t> mv(mParent, "MultiVector");
  mv.def(pybind11::init<pressiotools::py_f_arr>());
  mv.def(pybind11::init<pressiotools::py_c_arr>());
  mv.def(pybind11::init<typename v_t::ord_t, typename v_t::ord_t>());
  mv.def("data",	    &mv_t::data);
  mv.def("address",	    &mv_t::address);
  mv.def("extentLocal",	    &mv_t::extentLocal);
  mv.def("extentGlobal",    &mv_t::extentGlobal);
  mv.def("minRowGidLocal",  &mv_t::minRowGidLocal);
  mv.def("maxRowGidLocal",  &mv_t::maxRowGidLocal);
#ifndef PRESSIOTOOLS_ENABLE_TPL_MPI
  mv.def("extent",	    &mv_t::extent);
#endif

#ifdef PRESSIOTOOLS_ENABLE_TPL_TRILINOS
  // bind ops
  mParent.def("product", &pressiotools::ops::product);

  // bind tsqr
  using tsqr_t = pressiotools::Tsqr;
  pybind11::class_<tsqr_t> tsqr(mParent, "Tsqr");
  tsqr.def(pybind11::init());
  tsqr.def("computeThinOutOfPlace", &tsqr_t::computeThinOutOfPlace);
  tsqr.def("viewR",	 &tsqr_t::viewR,      pybind11::return_value_policy::reference);
  tsqr.def("viewQLocal", &tsqr_t::viewLocalQ, pybind11::return_value_policy::reference);

  // bind svd
  using svd_t = pressiotools::Svd;
  pybind11::class_<svd_t> svd(mParent, "Svd");
  svd.def(pybind11::init());
  svd.def("computeThin", &svd_t::computeThin);
  svd.def("viewSingValues",
	  &svd_t::viewS,  pybind11::return_value_policy::reference);
  svd.def("viewLeftSingVectorsLocal",
	  &svd_t::viewU,  pybind11::return_value_policy::reference);
  svd.def("viewRightSingVectorsT",
	  &svd_t::viewVT, pybind11::return_value_policy::reference);

  // bind pinv
  using pinv_t = pressiotools::Pinv;
  pybind11::class_<pinv_t> pinv(mParent, "PseudoInverse");
  pinv.def(pybind11::init());
  pinv.def("compute", &pinv_t::compute);
  pinv.def("viewTransposeLocal",
	   &pinv_t::viewLocalAstarT, pybind11::return_value_policy::reference);
  pinv.def("apply",
	   &pinv_t::apply,	     pybind11::return_value_policy::reference);
  pinv.def("applyTranspose",
	   &pinv_t::applyTranspose,  pybind11::return_value_policy::reference);
#endif
};

#endif
