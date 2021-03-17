
#ifndef PRESSIOTOOLS_PINV_HPP_
#define PRESSIOTOOLS_PINV_HPP_

#include "Epetra_MpiComm.h"
#include "Epetra_MultiVector.h"
#include "Epetra_Vector.h"

namespace pressiotools{

struct Pinv
{
  using mat_t = Teuchos::SerialDenseMatrix<int, pressiotools::scalar_t>;

  void compute(MultiVector & A)
  {
    // Let Apsi be the pseudo-inverse of A, i.e. Apsi = A^+
    //
    // NOTE that here we store (Apsi)^T because A is tall-skinny
    // and potentially with MANY rows, so we need to keep Apsi
    // to have the same row distribution as A

    // Apsi = V S^-1 U^T
    // BUT here we store: Apsi^T = U S^-1 V^T
    // where U, S and V are the SVD of A
    // and we assume S is a diagonal square matrix

    // References:
    // https://math.stackexchange.com/questions/19948/pseudoinverse-matrix-and-svd
    // https://www.johndcook.com/blog/2018/05/05/svd/

    if (A.extentGlobal(0) <= A.extentGlobal(1)){
      throw std::runtime_error
	("Parallel PseudoInverse: currently only works for tall skinny matrices.");
    }

    // compute SVD of A: A = U S V^T
    Svd svdObj;
    svdObj.computeThin(A);
    auto S = svdObj.viewSingularValues();
    const auto & U = svdObj.viewLocalLeftSingVectors();
    const auto & VT = svdObj.viewRightSingVectorsTransposed();
    const auto m = U.numRows();
    const auto n = U.numCols();

    // compute S^-1
    constexpr auto one  = static_cast<pressiotools::scalar_t>(1);
    if (Sinv_.shape(0) != n) Sinv_.resize({n}, false);
    for (int i=0; i<S.shape(0); ++i){
      Sinv_(i) = one/S(i);
    }

    // compute S^-1 V^T
    if (SinvVT_.numRows() != n or SinvVT_.numCols()!=n){
      SinvVT_.shape(n,n);
    }
    for (int j=0; j<SinvVT_.numCols(); ++j){
      for (int i=0; i<SinvVT_.numRows(); ++i){
	SinvVT_(i,j) += Sinv_(i) * VT(i,j);
      }
    }

    // compute U S^-1 V^T
    if (ApsiT_.numRows() != m or ApsiT_.numCols()!=n){
      ApsiT_.shape(m,n);
    }
    if (m != 0){
      constexpr auto zero = static_cast<pressiotools::scalar_t>(0);
      ApsiT_.multiply(Teuchos::NO_TRANS, Teuchos::NO_TRANS,
		      one, U, SinvVT_, zero);
    }
  }

  pressiotools::py_f_arr viewTransposedLocal(){
    pressiotools::py_f_arr view({ApsiT_.numRows(), ApsiT_.numCols()}, ApsiT_.values());
    return view;
  }

  pressiotools::py_f_arr apply(pressiotools::Vector & operand)
  {
    // note that here we store (A^+)^T so when
    // we need to compute the product: A^+ operand
    // we need to consider that

    using epmv_t = Epetra_MultiVector;
    using epv_t  = Epetra_Vector;

    Epetra_MpiComm comm(operand.communicator());
    const long long globalExtent = operand.extentGlobal();
    Epetra_Map map(globalExtent, 0, comm);

    // create epetra MV viewing (A^+)^T
    epmv_t epAsT(Epetra_DataAccess::View, map,
		 ApsiT_.values(),
		 ApsiT_.numRows(), ApsiT_.numCols());

    // create epetra vector viewing operand
    epv_t epOperand(Epetra_DataAccess::View, map,
		    operand.data().mutable_data());

    // compute A^* b
    auto numC = ApsiT_.numCols();
    pressiotools::py_f_arr result({numC});
    for (decltype(numC) i=0; i<numC; i++){
      int rc = epAsT(i)->Dot(epOperand, &result(i));
	if (rc!=0){
	  throw std::runtime_error("Error computing epetra dot");
	}
    }

    return result;
  }

  // here operand is 2d numpy array local to each rank
  pressiotools::py_f_arr applyTranspose(pressiotools::py_f_arr operand)
  {
    if(operand.ndim() !=2 ){
      throw std::runtime_error
	("pvind::applyTranspose: invalid call for operand rank!=2");
    }

    if(ApsiT_.numCols() != operand.shape(0) ){
      throw std::runtime_error
	("pvind::applyTranspose: non-matching extent");
    }

    // we need to make this more efficient, but not now use this
    const int opNumCols = operand.shape(1);
    pressiotools::py_f_arr result({ApsiT_.numRows(), opNumCols});
    for (int i=0; i<result.shape(0); ++i)
    {
      for (int j=0; j<result.shape(1); ++j)
      {
	pressiotools::scalar_t tmp = 0;
	for (int k=0; k<ApsiT_.numCols(); ++k){
	  tmp += ApsiT_(i,k) * operand(k,j);
	}
	result(i,j) = tmp;
      }
    }

    return result;
  }

private:
  pressiotools::py_f_arr Sinv_;

  // let Apsi be the pseudo-inverse of A, i.e. Apsi = A^+
  // here we store ApsiT = transpose(Apsi) as explained at top
  mat_t ApsiT_;

  // to store S^-1 V^T
  mat_t SinvVT_;
};

}//end namespace pressiotools
#endif
