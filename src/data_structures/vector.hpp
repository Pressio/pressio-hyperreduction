
#ifndef PRESSIOTOOLS_DS_VEC_HPP_
#define PRESSIOTOOLS_DS_VEC_HPP_

#include <iostream>

namespace pressiotools{

class Vector
{
  py_f_arr data_;
  MPI_Comm comm_ = MPI_COMM_WORLD;
  std::size_t rank_ = 0;
  std::size_t globalExtent_ = {};

public:
  Vector() = delete;

  Vector(pybind11::ssize_t localextent)
    : data_(localextent)
  {
    constexpr auto zero = static_cast<pressiotools::scalar_t>(0);
    for (pybind11::ssize_t i=0; i<localextent; ++i) data_(i) = zero;

    computeGlobalExtent();
  }

  Vector(py_f_arr dataIn) : data_(dataIn){
    if (dataIn.ndim() != 1 ){
      throw std::runtime_error
	("A vector is only constructible from rank-1 array");
    }

    computeGlobalExtent();
  }

  Vector(py_c_arr dataIn) : data_(dataIn){
    if (dataIn.ndim() != 1 ){
      throw std::runtime_error
	("A vector is only constructible from rank-1 array");
    }

    computeGlobalExtent();
  }

  Vector(const Vector & other) = default;
  Vector & operator=(const Vector & other) = default;
  Vector(Vector && other) = default;
  Vector & operator=(Vector && other) = default;
  ~Vector() = default;

  const MPI_Comm & communicator() const{
    return comm_;
  }

  pybind11::ssize_t extentLocal() const{
    return data_.shape(0);
  }

  pybind11::ssize_t extentGlobal() const{
    return globalExtent_;
  }

  py_f_arr data(){
    return data_;
  }

  pressiotools::scalar_t sumGlobal() const
  {
    constexpr auto zero = static_cast<pressiotools::scalar_t>(0);
    pressiotools::scalar_t localsum = zero;
    pressiotools::scalar_t globalsum = zero;

    for (pybind11::ssize_t i=0; i<extentLocal(); ++i) localsum += data_(i);

    MPI_Allreduce(&localsum, &globalsum, 1, MPI_DOUBLE, MPI_SUM, comm_);
    return globalsum;
  }

private:
  void computeGlobalExtent()
  {
    const std::size_t localN = data_.shape(0);
    MPI_Allreduce(&localN, &globalExtent_, 1,
		  MPI_UNSIGNED_LONG, MPI_SUM, comm_);
  }
};

}//end namespace pressiotools
#endif
