
#ifndef PRESSIOTOOLS_DS_VEC_HPP_
#define PRESSIOTOOLS_DS_VEC_HPP_

#include "mpi.h"

namespace pressiotools{

class Vector
{
  py_f_arr data_;
  MPI_Comm comm_ = MPI_COMM_WORLD;
  std::size_t rank_ = 0;
  std::size_t globalExtent_ = {};

public:
  Vector() = delete;
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

  std::size_t extentLocal() const{
    return data_.shape(0);
  }

  std::size_t extentGlobal() const{
    return globalExtent_;
  }

  py_f_arr data(){
    return data_;
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
