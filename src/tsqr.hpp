
#ifndef PRESSIOTOOLS_TSQR_HPP_
#define PRESSIOTOOLS_TSQR_HPP_

#include "Epetra_MpiComm.h"
#include "Epetra_MultiVector.h"
#include "Epetra_TsqrAdaptor.hpp"
#include <Epetra_Import.h>
#include <cmath>

namespace pressiotools{

struct Tsqr
{
  using int_t		  = int;
  using tsqr_adaptor_type = Epetra::TsqrAdaptor;
  using R_t	= Teuchos::SerialDenseMatrix<int_t, scalar_t>;
  using R_ptr_t	= std::shared_ptr<R_t>;
  using mv_t	= Epetra_MultiVector;
  using Q_ptr_t	= std::shared_ptr<mv_t>;

  // we need to non-const ref A to do the computation
  // but we do NOT modify A here
  void computeThinOutOfPlace(MultiVector & A)
  {
    if (A.extentGlobal(0) <= A.extentGlobal(1)){
      throw std::runtime_error
	("Parallel TSQR currently only supports tall skinny matrices.");
    }

    // check if A needs to be redistributed before computing QR
    // and pick the implementation accordingly
    if (this->needsRedistribution(A)){
      redistributeAndComputeThinImpl(A);
    }
    else{
      computeThinOutOfPlaceImpl(A);
    }
  }

  // note that this does not allocate anything,
  // it just views the data in pybind::array to pass it to Python
  pressiotools::py_f_arr viewR()
  {
    pressiotools::py_f_arr Rview({R_->numRows(), R_->numCols()},
				 R_->values());
    return Rview;
  }

  // note that this does not allocate anything,
  // it just views the data in pybind::array to pass it to Python
  pressiotools::py_f_arr viewLocalQ()
  {
    pressiotools::py_f_arr Qview({Q_->MyLength(), Q_->NumVectors()},
				 Q_->Values());
    return Qview;
  }

private:
  void redistributeAndComputeThinImpl(MultiVector & A)
  {
    /*
      we need to first redistribute A and then compute TSQR.
      This is needed because for TSQR to work, on rach rank
      participating to tsqr, the num of rows must be >= num of cols.
      So if A does not meet this condition, we redistribute it
      to make sure that condition is met, then perform Tsqr and then
      redistribute the results back.
    */

    auto & comm = A.communicator();
    int_t rank = 0;
    int_t commSize = 0;
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

    // view A inside an epetra::MV
    // here we do not allocate new mem, we just view A
    Epetra_MpiComm commE(comm);
    Epetra_Map mapA(globNumRows, A.extentLocal(0), 0, commE);
    scalar_t * ptr = A.data().mutable_data();
    mv_t epA(Epetra_DataAccess::View, mapA, ptr, A.extentLocal(0), globNumCols);
    //epA.Print(std::cout);

    if (maxNumRank >= commSize)
    {
      // this case means that some ranks have too few rows of A to
      // satisfy the tsqr requirement, but if we were to distribute
      // A **uniformly** over the same communicator, we can still use
      // all ranks since they all meet the TSQR condition.

      // **uniformly** distribute A into A2
      Epetra_Map newMap(A.extentGlobal(0), 0, commE);
      mv_t A2(newMap, globNumCols);
      Epetra_Import imp(newMap, mapA);
      A2.Import(epA, imp, Insert);
      //A2.Print(std::cout);

      // compute A2 = tmpQ R
      // tmpQ will have the same row-distribution as A2 but
      // will have to redistribute tmpQ to the Q of the original A
      // since our goal is to compute the QR of A
      // note that while Q needs to be redistributed,
      // R remains the same as the R of the orignal A
      createRIfNeeded(globNumCols);
      mv_t tmpQ(newMap, globNumCols);
      constexpr auto zero = static_cast<pressiotools::scalar_t>(0);
      tmpQ.PutScalar(zero);
      R_->putScalar(zero);
      tsqr_adaptor_type tsqrAdaptor;
      tsqrAdaptor.factorExplicit(A2, tmpQ, *R_, false);

      // now I need to redistribute Q to match original A
      createQIfNeeded(mapA, globNumCols);
      Q_->PutScalar(zero);
      Epetra_Import imp2(mapA, newMap);
      Q_->Import(tmpQ, imp2, Insert);
      //Q_->Print(std::cout);
    }
    else
    {
      // this branch means that even if we redistribute A
      // uniformly over the same comm, some ranks would end up with zero entries.
      // so here maxNumRank corresponds to the max # of ranks
      // needed to meet the conditions for Tsqr.
      // We redistribute A over maxNumRanks, and these ranks
      // compute the tsqr, but this also means that some ranks in the
      // original comm will be left out, so we need to construct
      // a new subcomm to do all this.

      // every rank < maxNumRank has as many rows as cols,
      // except for rank==maxNumRank-1 which will have that plus
      // all the rows neededd to compensate for remainder.
      // All ranks > maxNumRank do not participate the tsqr so have zero data.
      int_t myNumRows = 0;
      if (rank < maxNumRank-1){
	myNumRows = globNumCols;
      }
      else if (rank == maxNumRank-1){
	myNumRows = globNumCols + (globNumRows % globNumCols);
      }

      // create a A2
      Epetra_Map A2map(A.extentGlobal(0), myNumRows, 0, commE);
      mv_t A2(A2map, globNumCols);
      Epetra_Import imp(A2map, mapA);
      A2.Import(epA, imp, Insert);
      mv_t newQ(A2map, globNumCols);

      // create a new comm for just the subset of ranks
      // that will compute the QR
      int color = (rank < maxNumRank) ? 1 : 0;
      MPI_Comm subComm;
      int rc = MPI_Comm_split(comm, color, 0, &subComm);
      if (rc != MPI_SUCCESS){
	throw std::runtime_error("Error splitting comm in TSQR");
      }
      int rank2;
      MPI_Comm_rank(subComm, &rank2);

      // need to create the R factor, recall that R computed from A2
      // is the same as the R factoring the original A
      constexpr auto zero = static_cast<pressiotools::scalar_t>(0);
      createRIfNeeded(globNumCols);
      R_->putScalar(zero);
      if (color==1)
      {
	Epetra_MpiComm subCommE(subComm);
	Epetra_Map A3map(A2map.NumGlobalElements(), myNumRows, 0, subCommE);
	scalar_t * ptr = A2.Values();
	mv_t A3(Epetra_DataAccess::View, A3map, ptr,
		A3map.NumMyElements(), globNumCols);
	//A3.Print(std::cout);

	// create the tmpQ to store the left-sing vecs of newA
	mv_t tmpQ(A3map, globNumCols);
	tmpQ.PutScalar(zero);
	tsqr_adaptor_type tsqrAdaptor;
	tsqrAdaptor.factorExplicit(A3, tmpQ, *R_, false);

	if (A2map.NumMyElements() != A3map.NumMyElements()){
	  throw std::runtime_error("Non matching num local elements, something wrong");
	}

	for (int_t j=0; j<globNumCols; ++j){
	  for (int_t i=0; i<A3map.NumMyElements(); ++i){
	    newQ[j][i] = tmpQ[j][i];
	  }
	}
      }

      // we need to broadcast R since only a subset of ranks
      // compute it but all of them should have it
      commE.Broadcast(R_->values(), globNumCols*globNumCols, 0);
      commE.Barrier();

      // now I need to redistribute tmpQ to match original A
      // note that some ranks might have zero rows of Q
      // which is fine if that happens
      createQIfNeeded(mapA, globNumCols);
      Q_->PutScalar(static_cast<pressiotools::scalar_t>(0));
      Epetra_Import imp2(mapA, A2map);
      Q_->Import(newQ, imp2, Insert);
      //Q_->Print(std::cout);

      commE.Barrier();
    }
  }

  void computeThinOutOfPlaceImpl(MultiVector & A)
  {
    // this is the impl when A is suitable for Tsqr

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
    // if at least one rank has: local num of rows < num cols,
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

    return *(std::max_element(allFlags.cbegin(), allFlags.cend())) == 1;
  }

private:
  Q_ptr_t Q_ = {};
  R_ptr_t R_ = {};
};

}//end namespace pressiotools
#endif
