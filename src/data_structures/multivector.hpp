
#ifndef PRESSIOTOOLS_DS_MV_HPP_
#define PRESSIOTOOLS_DS_MV_HPP_

namespace pressiotools{

class MultiVector
{
  py_f_arr data_;
#ifdef PRESSIOTOOLS_ENABLE_TPL_MPI
  MPI_Comm comm_ = MPI_COMM_WORLD;
#endif
  std::size_t rank_ = 0;

  using shape_t = std::array<std::size_t, 2>;
  shape_t globalShape_ = {};

public:
  MultiVector() = delete;

  MultiVector(py_f_arr dataIn) : data_(dataIn){
    if (dataIn.ndim() != 2 ){
      throw std::runtime_error
	("A MultiVector is only constructible from rank-2 array");
    }

    computeGlobalNumRows();
    globalShape_[1] = data_.shape(1);
  }

  MultiVector(py_c_arr dataIn){
    throw std::runtime_error
      ("You cannot construct a MultiVector from a row-major numpy array. \
You can use np.asfortranarray() to make an array column-major.");
  }

  MultiVector(const MultiVector & other) = default;
  MultiVector & operator=(const MultiVector & other) = default;
  MultiVector(MultiVector && other) = default;
  MultiVector & operator=(MultiVector && other) = default;
  ~MultiVector() = default;

#ifdef PRESSIOTOOLS_ENABLE_TPL_MPI
  const MPI_Comm & communicator() const{
    return comm_;
  }
#endif

  std::size_t extentLocal(int i) const{
    return data_.shape(i);
  }

  std::size_t extentGlobal(int i) const{
    return globalShape_[i];
  }

#ifndef PRESSIOTOOLS_ENABLE_TPL_MPI
  std::size_t extent(int i) const{
    return globalShape_[i];
  }
#endif

  py_f_arr data(){
    return data_;
  }

private:
  void computeGlobalNumRows()
  {
    const std::size_t localN = data_.shape(0);
    globalShape_[0]=localN;
#ifdef PRESSIOTOOLS_ENABLE_TPL_MPI
    MPI_Allreduce(&localN, &globalShape_[0], 1,
		  MPI_UNSIGNED_LONG, MPI_SUM, comm_);
#endif
  }
};

}//end namespace pressiotools
#endif
