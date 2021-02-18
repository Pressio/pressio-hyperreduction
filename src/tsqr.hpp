
#ifndef PRESSIOTOOLS_TSQR_HPP_
#define PRESSIOTOOLS_TSQR_HPP_

#include "Epetra_MpiComm.h"
#include "Epetra_MultiVector.h"
#include "Epetra_TsqrAdaptor.hpp"
#include <Teuchos_LAPACK.hpp>

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
    if (A.extentLocal(0) < A.extentLocal(1)){
      throw std::runtime_error
	("The input matrix must have at least as many rows on each processor as there are columns.");
    }

    const int nRowLocal = A.extentLocal(0);
    const long long int nRowGlobal = A.extentGlobal(0);
    const long long int nColGlobal = A.extentGlobal(1);

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

    tsqrAdaptor_.factorExplicit(epA, *Q_, *R_, false);
    //Q_->Print(std::cout << std::setprecision(10));
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

private:
  tsqr_adaptor_type tsqrAdaptor_;
  Q_ptr_t Q_ = {};
  R_ptr_t R_ = {};
};

}//end namespace pressiotools
#endif
