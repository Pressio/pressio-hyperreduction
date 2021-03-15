
#ifndef PRESSIOTOOLS_DS_MV_HPP_
#define PRESSIOTOOLS_DS_MV_HPP_

#include <cstdint>

namespace pressiotools{

class MultiVector
{
public:
  using ord_t = int;


public:
  MultiVector() = delete;

  MultiVector(const ord_t locExt0, const ord_t locExt1)
    : data_({locExt0, locExt1})
  {
    constexpr auto zero = static_cast<pressiotools::scalar_t>(0);
    for (ord_t j=0; j<locExt1; ++j){
      for (ord_t i=0; i<locExt0; ++i){
	data_(i,j) = zero;
      }
    }

#ifdef PRESSIOTOOLS_ENABLE_TPL_MPI
    MPI_Comm_rank(comm_, &rank_);
#endif
    computeGlobalNumRows();
    globalShape_[1] = data_.shape(1);
    computeGlobalIDsRange();
  }

  MultiVector(py_f_arr dataIn) : data_(dataIn)
  {
    if (dataIn.ndim() != 2 ){
      throw std::runtime_error
	("A MultiVector is only constructible from rank-2 array");
    }

#ifdef PRESSIOTOOLS_ENABLE_TPL_MPI
    MPI_Comm_rank(comm_, &rank_);
#endif
    computeGlobalNumRows();
    globalShape_[1] = data_.shape(1);
    computeGlobalIDsRange();
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

  ord_t minRowGidLocal() const{
    return myGidRange_[0];
  }

  ord_t maxRowGidLocal() const{
    return myGidRange_[1];
  }

  ord_t extentLocal(int i) const{
    return data_.shape(i);
  }

  ord_t extentGlobal(int i) const{
    return globalShape_[i];
  }

#ifndef PRESSIOTOOLS_ENABLE_TPL_MPI
  ord_t extent(int i) const{
    return globalShape_[i];
  }
#endif

  py_f_arr data(){
    return data_;
  }

  uintptr_t address() const{
    return reinterpret_cast<uintptr_t>(data_.data());
  }

private:
  void computeGlobalNumRows()
  {
    const ord_t localN = data_.shape(0);
    globalShape_[0]=localN;
#ifdef PRESSIOTOOLS_ENABLE_TPL_MPI
    MPI_Allreduce(&localN, &globalShape_[0], 1,
		  MPI_UNSIGNED_LONG, MPI_SUM, comm_);
#endif
  }

  void computeGlobalIDsRange()
  {
    ord_t localN = data_.shape(0);
#ifdef PRESSIOTOOLS_ENABLE_TPL_MPI

    ord_t commSize;
    MPI_Comm_size(comm_, &commSize);
    std::vector<ord_t> allNs(commSize);
    MPI_Allgather(&localN, 1, MPI_INT,
		  allNs.data(), 1,
		  MPI_INT, comm_);

    if (rank_==0)
    {
      myGidRange_ = {0, localN-1};
    }
    else{
      ord_t sum = 0;
      for (ord_t i=0; i<=rank_-1; ++i){
	sum += allNs[i];
      }
      myGidRange_ = {sum, sum+localN-1};
    }

#else
    myGidRange_ = {0, localN-1};
#endif
  }

private:
#ifdef PRESSIOTOOLS_ENABLE_TPL_MPI
  MPI_Comm comm_ = MPI_COMM_WORLD;
#endif
  py_f_arr data_;
  ord_t rank_ = 0;
  std::array<ord_t, 2> globalShape_ = {};
  std::array<ord_t, 2> myGidRange_ = {};

};

}//end namespace pressiotools
#endif
