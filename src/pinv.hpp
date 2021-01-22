
#ifndef PRESSIOTOOLS_PINV_HPP_
#define PRESSIOTOOLS_PINV_HPP_

#include "Epetra_MpiComm.h"
#include "Epetra_MultiVector.h"
#include "Epetra_Vector.h"

namespace pressiotools{

struct Pinv
{
  using mat_t = Teuchos::SerialDenseMatrix<int, scalar_t>;

  void compute(MultiVector & A)
  {
    constexpr auto zero = static_cast<pressiotools::scalar_t>(0);
    constexpr auto one  = static_cast<pressiotools::scalar_t>(1);

    svd_.computeThin(A);
    auto S = svd_.viewS();
    const auto & U = svd_.viewNativeU();
    const auto & VT = svd_.viewNativeVT();
    const auto m = U.numRows();
    const auto n = U.numCols();

    if (Sinv_.shape(0) != n) Sinv_.resize({n}, false);
    for (int i=0; i<S.shape(0); ++i){
      Sinv_(i) = one/S(i);
      //std::cout << Sinv_(i) << std::endl;
    }

    // compute SinvVT =  S* times V^T
    if (SinvVT_.numRows() != n or SinvVT_.numCols()!=n)
      SinvVT_.shape(n,n);

    for (int j=0; j<SinvVT_.numCols(); ++j){
      for (int i=0; i<SinvVT_.numRows(); ++i){
	SinvVT_(i,j) += Sinv_(i) * VT(i,j);
      }
    }

    // for (int i=0; i<SinvVT_.numRows(); ++i){
    //   for (int j=0; j<SinvVT_.numCols(); ++j){
    // 	std::cout <<SinvVT_(i,j) << " ";
    //   }
    //   std::cout << std::endl;
    // }

    // compute A*T =  U times S* times V^T
    if (AsT_.numRows() != m or AsT_.numCols()!=n)
      AsT_.shape(m,n);

    AsT_.multiply(Teuchos::NO_TRANS, Teuchos::NO_TRANS,
		  one, U, SinvVT_, zero);
  }

  pressiotools::py_f_arr viewLocalAstarT(){
    pressiotools::py_f_arr view({AsT_.numRows(), AsT_.numCols()}, AsT_.values());
    return view;
  }

  pressiotools::py_f_arr apply(pressiotools::Vector & operand)
  {
    // note that pinv does not store A^* but A^*T so when
    // we need to apply A^* we need to consider that

    using epmv_t = Epetra_MultiVector;
    using epv_t  = Epetra_Vector;

    Epetra_MpiComm comm(operand.communicator());
    const long long globalExtent = operand.extentGlobal();
    Epetra_Map map(globalExtent, 0, comm);

    // create epetra MV viewing A^*T
    epmv_t epAsT(Epetra_DataAccess::View, map,
		 AsT_.values(),
		 AsT_.numRows(), AsT_.numCols());

    // create epetra vector viewing operand
    epv_t epOperand(Epetra_DataAccess::View, map, operand.data().mutable_data());

    // compute A^* b
    auto numC = AsT_.numCols();
    pressiotools::py_f_arr result({numC});
    for (decltype(numC) i=0; i<numC; i++){
      epAsT(i)->Dot(epOperand, &result(i));
    }

    return result;
  }

private:
  Svd svd_;
  pressiotools::py_f_arr Sinv_;
  mat_t AsT_;
  mat_t SinvVT_;
};

}//end namespace pressiotools
#endif
