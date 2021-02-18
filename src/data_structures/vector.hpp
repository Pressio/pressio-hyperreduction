
#ifndef PRESSIOTOOLS_DS_VEC_HPP_
#define PRESSIOTOOLS_DS_VEC_HPP_

namespace pressiotools{

class Vector
{
  py_f_arr data_;
#ifdef PRESSIOTOOLS_ENABLE_TPL_MPI
  MPI_Comm comm_ = MPI_COMM_WORLD;
#endif
  std::size_t rank_ = 0;
  std::size_t globalExtent_ = {};

public:
  Vector() = delete;

  Vector(pybind11::ssize_t localextent)
    : data_(localextent)
  {
    constexpr auto zero = static_cast<pressiotools::scalar_t>(0);
    for (pybind11::ssize_t i=0; i<localextent; ++i){
      data_(i) = zero;
    }

    computeGlobalExtent();
  }

  Vector(py_f_arr dataIn)
    : data_(dataIn)
  {
    if (dataIn.ndim() != 1 ){
      throw std::runtime_error
	("A vector is only constructible from rank-1 array");
    }

    computeGlobalExtent();
  }

  Vector(py_c_arr dataIn)
    : data_(dataIn)
  {
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

  py_f_arr data(){
    return data_;
  }

#ifdef PRESSIOTOOLS_ENABLE_TPL_MPI
  const MPI_Comm & communicator() const{
    return comm_;
  }
#endif

  pybind11::ssize_t extentLocal() const{
    return data_.shape(0);
  }

  pybind11::ssize_t extentGlobal() const{
    return globalExtent_;
  }

  pressiotools::scalar_t sumLocal() const
  {
    constexpr auto zero = static_cast<pressiotools::scalar_t>(0);
    pressiotools::scalar_t localsum = zero;
    for (pybind11::ssize_t i=0; i<extentLocal(); ++i){
      localsum += data_(i);
    }
    return localsum;
  }

  pressiotools::scalar_t sumGlobal() const
  {
    static_assert
      (std::is_same<pressiotools::scalar_t, double>::value,
       "Method only suitable for scalar= double");

#ifdef PRESSIOTOOLS_ENABLE_TPL_MPI
    constexpr auto zero = static_cast<pressiotools::scalar_t>(0);
    const auto localsum = sumLocal();
    pressiotools::scalar_t globalsum = zero;
    MPI_Allreduce(&localsum, &globalsum, 1, MPI_DOUBLE, MPI_SUM, comm_);
    return globalsum;
#else
    return sumLocal();
#endif
  }

#ifndef PRESSIOTOOLS_ENABLE_TPL_MPI
  pybind11::ssize_t extent() const{
    return data_.shape(0);
  }

  pressiotools::scalar_t sum() const
  {
    return sumLocal();
  }
#endif

private:
  void computeGlobalExtent()
  {
    const std::size_t localN = data_.shape(0);

#ifdef PRESSIOTOOLS_ENABLE_TPL_MPI
    MPI_Allreduce(&localN, &globalExtent_, 1,
		  MPI_UNSIGNED_LONG, MPI_SUM, comm_);
#else
    globalExtent_ = localN;
#endif
  }
};

}//end namespace pressiotools
#endif
