
#ifndef PRESSIOTOOLS_TSQR_HPP_
#define PRESSIOTOOLS_TSQR_HPP_

#include "Epetra_MpiComm.h"
#include "Epetra_MultiVector.h"
#include "Epetra_TsqrAdaptor.hpp"
#include <Teuchos_LAPACK.hpp>
#include <Epetra_Import.h>
#include <cmath>

// if (A.extentLocal(0) < A.extentLocal(1)){
//   throw std::runtime_error
// 	("The input matrix must have at least as many rows on each processor as there are columns.");
// }

namespace pressiotools{

struct Tsqr
{
  using int_t		  = int;
  using tsqr_adaptor_type = Epetra::TsqrAdaptor;
  using R_t	= Teuchos::SerialDenseMatrix<int_t, scalar_t>;
  using R_ptr_t	= std::shared_ptr<R_t>;
  using mv_t	= Epetra_MultiVector;
  using Q_ptr_t	= std::shared_ptr<mv_t>;

  void computeThinOutOfPlace(MultiVector & A)
  {
    if (A.extentGlobal(0) <= A.extentGlobal(1)){
      throw std::runtime_error
	("Parallel TSQR: currently only supports tall skinny matrices.");
    }

    const auto redistribute = this->needsRedistribution(A);
    if (redistribute){
      redistributeAndComputeThinImpl(A);
    }
    else{
      computeThinOutOfPlaceImpl(A);
    }
  }

  // note that this does not allocate anything, it views the data
  // inside a pybind::array so that we can pass it to Python
  pressiotools::py_f_arr viewR()
  {
    pressiotools::py_f_arr Rview({R_->numRows(), R_->numCols()},
				 R_->values());
    return Rview;
  }

  // note that this does not allocate anything, it views the data
  // inside a pybind::array so that we can pass it to Python
  pressiotools::py_f_arr viewLocalQ()
  {
    pressiotools::py_f_arr Qview({Q_->MyLength(), Q_->NumVectors()},
				 Q_->Values());
    return Qview;
  }

private:
  void redistributeAndComputeThinImpl(MultiVector & A)
  {
    auto & comm = A.communicator();
    int_t rank;
    int_t commSize;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &commSize);

    // find the max num of ranks that can be used to redistribute
    // the matrix satisfying such that each rank has # rows >= cols
    const int_t globNumRows = A.extentGlobal(0);
    const int_t globNumCols = A.extentGlobal(1);
    const int_t maxNumRank = (int) std::floor(globNumRows/globNumCols);
    // std::cout << globNumRows << " "
    // 		<< globNumCols << " "
    // 		<< maxNumRank << std::endl;

    Epetra_MpiComm commE(comm);
    // create epetra MV for current A
    Epetra_Map mapA(globNumRows, A.extentLocal(0), 0, commE);
    scalar_t * ptr = A.data().mutable_data();
    mv_t epA(Epetra_DataAccess::View, mapA, ptr, A.extentLocal(0), globNumCols);
    //epA.Print(std::cout);

    if (maxNumRank >= commSize)
    {
      // this case means that some ranks had too few rows of A to
      // satisfy the tsqr requirement, but if we were to distribute
      // A **uniformly** over the same communicator, we can still use
      // all ranks and meet the TSQR condition.

      // **uniformly** distribute A into newA
      Epetra_Map newMap(A.extentGlobal(0), 0, commE);
      mv_t newA(newMap, globNumCols);
      Epetra_Import imp(newMap, mapA);
      newA.Import(epA, imp, Insert);
      //newA.Print(std::cout);

      // to compute QR of new A, R is the same as the orignal
      // but Q will have the same distribution as newA so I need
      // to create a tmpQ to use for the newA and then
      // redistributed tmpQ into the Q that would fit oriignal A
      createRIfNeeded(globNumCols);
      mv_t tmpQ(newMap, globNumCols);
      constexpr auto zero = static_cast<pressiotools::scalar_t>(0);
      tmpQ.PutScalar(zero);
      R_->putScalar(zero);
      tsqr_adaptor_type tsqrAdaptor;
      tsqrAdaptor.factorExplicit(newA, tmpQ, *R_, false);

      // now I need to redistribute Q to match original A
      createQIfNeeded(mapA, globNumCols);
      Q_->PutScalar(zero);
      Epetra_Import imp2(mapA, newMap);
      Q_->Import(tmpQ, imp2, Insert);
      //Q_->Print(std::cout);
    }
    else
    {
      // this branching means that even if we redistribute A
      // uniformly, some ranks would end up with zero entries

      int_t myN = 0;
      if (rank < maxNumRank-1){
	myN = globNumCols;
      }
      else if (rank == maxNumRank-1){
	myN = globNumCols + (globNumRows % globNumCols);
      }

      //std::cout << "here: " << rank << " " << myN  << std::endl;
      Epetra_Map newMap(A.extentGlobal(0), myN, 0, commE);
      mv_t newA(newMap, globNumCols);
      Epetra_Import imp(newMap, mapA);
      newA.Import(epA, imp, Insert);
      mv_t newQ(newMap, globNumCols);

      // create a new comm for just the subset of ranks
      // that will compute the QR
      int color = (rank < maxNumRank) ? 1 : 0;
      MPI_Comm subComm;
      MPI_Comm_split(comm, color, 0, &subComm);
      int rank2;
      int rc = MPI_Comm_rank(subComm, &rank2);
      assert(rc == MPI_SUCCESS);
      //std::cout << rank << " " << rank2 << std::endl;

      constexpr auto zero = static_cast<pressiotools::scalar_t>(0);
      createRIfNeeded(globNumCols);
      R_->putScalar(zero);
      if (color==1)
      {
	Epetra_MpiComm comm2E(subComm);
	Epetra_Map newMap2(newMap.NumGlobalElements(), myN, 0, comm2E);
	scalar_t * ptr = newA.Values();
	mv_t newA2(Epetra_DataAccess::View, newMap2, ptr,
		   newMap2.NumMyElements(), globNumCols);
	//newA2.Print(std::cout);

	mv_t tmpQ(newMap2, globNumCols);
	tmpQ.PutScalar(zero);
	tsqr_adaptor_type tsqrAdaptor;
	tsqrAdaptor.factorExplicit(newA2, tmpQ, *R_, false);

	if (newMap.NumMyElements() != newMap2.NumMyElements()){
	  throw std::runtime_error("Non matching num local elements, something wrong");
	}

	for (int_t j=0; j<globNumCols; ++j){
	  for (int_t i=0; i<newMap2.NumMyElements(); ++i){
	    newQ[j][i] = tmpQ[j][i];
	  }
	}
      }
      // we need to broacast R since only a subset of ranks compute it
      // but all of them should have it
      commE.Broadcast(R_->values(), globNumCols*globNumCols, 0);
      commE.Barrier();

      // now I need to redistribute Q to match original A
      createQIfNeeded(mapA, globNumCols);
      Q_->PutScalar(static_cast<pressiotools::scalar_t>(0));
      Epetra_Import imp2(mapA, newMap);
      Q_->Import(newQ, imp2, Insert);
      //Q_->Print(std::cout);

      commE.Barrier();
    }
  }

  void computeThinOutOfPlaceImpl(MultiVector & A)
  {
    const int_t nRowLocal  = A.extentLocal(0);
    const int_t nRowGlobal = A.extentGlobal(0);
    const int_t nColGlobal = A.extentGlobal(1);

    Epetra_MpiComm comm(A.communicator());
    Epetra_Map map(nRowGlobal, nRowLocal, 0, comm);

    // create epetra MV viewing A
    scalar_t * ptr = A.data().mutable_data();
    mv_t epA(Epetra_DataAccess::View, map, ptr, A.extentLocal(0), nColGlobal);
    createRIfNeeded(nColGlobal);
    createQIfNeeded(map, nColGlobal);
    constexpr auto zero = static_cast<pressiotools::scalar_t>(0);
    Q_->PutScalar(zero);
    R_->putScalar(zero);

    tsqr_adaptor_type tsqrAdaptor;
    tsqrAdaptor.factorExplicit(epA, *Q_, *R_, false);
    //Q_->Print(std::cout << std::setprecision(10));
  }

  void createRIfNeeded(int_t n)
  {
    if (!R_ or (R_->numRows()!=n and R_->numCols()!=n)){
      R_ = std::make_shared<R_t>(n, n);
    }
  }

  template <typename map_t>
  void createQIfNeeded(const map_t & map, int_t cols)
  {
    if (!Q_ or !Q_->Map().SameAs(map))
      Q_ = std::make_shared<mv_t>(map, cols);
  }

  bool needsRedistribution(const MultiVector & A) const
  {
    // if for at least one rank we have that
    // local num of rows < num cols,
    // we cannot use directly the trilinos tsqr
    // we need to redistribute the matrix so that we can use TSQR

    // flag = 1 if this rank does not meet condition
    const int flag = A.extentLocal(0) < A.extentLocal(1) ? 1 : 0;

    auto & comm_ = A.communicator();

    int_t rank;
    MPI_Comm_rank(comm_, &rank);
    int_t commSize;
    MPI_Comm_size(comm_, &commSize);
    //std::cout << rank << " " << flag << std::endl;

    std::vector<int_t> allFlags(commSize);
    MPI_Allgather(&flag, 1, MPI_INT,
		  allFlags.data(), 1,
		  MPI_INT, comm_);

    // if (rank==0){
    //   for (auto it : allFlags)
    // 	std::cout << it << " ";
    //   std::cout << std::endl;
    // }
    return *(std::max_element(allFlags.cbegin(), allFlags.cend())) == 1;
  }

private:
  Q_ptr_t Q_ = {};
  R_ptr_t R_ = {};
};

}//end namespace pressiotools
#endif
