
#ifndef PRESSIOTOOLS_DS_MV_HPP_
#define PRESSIOTOOLS_DS_MV_HPP_

namespace pressiotools{

class MultiVector
{
  py_f_arr data_;
#ifdef PRESSIOTOOLS_ENABLE_TPL_MPI
  MPI_Comm comm_ = MPI_COMM_WORLD;
#endif
  int rank_ = 0;

  std::array<std::size_t, 2> globalShape_ = {};
  std::array<std::size_t, 2> myGidRange_ = {};

public:
  MultiVector() = delete;

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

  std::size_t minRowGidLocal() const{
    return myGidRange_[0];
  }

  std::size_t maxRowGidLocal() const{
    return myGidRange_[1];
  }

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

  void computeGlobalIDsRange()
  {
    std::size_t localN = data_.shape(0);
#ifdef PRESSIOTOOLS_ENABLE_TPL_MPI

    int commSize;
    MPI_Comm_size(comm_, &commSize);
    std::vector<std::size_t> allNs(commSize);
    MPI_Allgather(&localN, 1, MPI_UNSIGNED_LONG,
		  allNs.data(), 1,
		  MPI_UNSIGNED_LONG, comm_);

    if (rank_==0)
    {
      myGidRange_ = {0, localN-1};
    }
    else{
      std::size_t sum = 0;
      for (std::size_t i=0; i<=rank_-1; ++i){
	sum += allNs[i];
      }
      myGidRange_ = {sum, sum+localN-1};
    }

#else
    myGidRange_ = {0, localN-1};
#endif
  }
};

}//end namespace pressiotools
#endif
