
#ifndef PRESSIOTOOLS_SVD_HPP_
#define PRESSIOTOOLS_SVD_HPP_

#include <Teuchos_LAPACK.hpp>

namespace pressiotools{

struct Svd
{
  using mat_t = Teuchos::SerialDenseMatrix<int, scalar_t>;

  void computeThin(MultiVector & A)
  {
    //http://www.netlib.org/lapack/explore-html/d1/d7e/group__double_g_esing_ga84fdf22a62b12ff364621e4713ce02f2.html
    //https://spec.oneapi.com/versions/0.5.0/oneMKL/GUID-9753DBC0-04D2-40EA-B1BD-4C83D1FD8C43.html
    //https://docs.trilinos.org/dev/packages/teuchos/doc/html/classTeuchos_1_1LAPACK.html#a8a8a8168153fc043d900541d745a850d

    Tsqr qrObj;
    Teuchos::LAPACK<int, pressiotools::scalar_t> lpkObj;

    if (A.extentGlobal(0) <= A.extentGlobal(1)){
      throw std::runtime_error
	("Parallel SVD: currently only accepts a tall skinny matrices.");
    }

    // -------------------------------------
    // 1. compute A = QR
    qrObj.computeThinOutOfPlace(A);

    // view the factors (note that these have view semantics)
    // R is replicated on each rank, it is a square and small matrix
    // Q is distributed, we viewing the local part only views the
    // rows that belong to this rank
    auto R = qrObj.viewR();
    auto Q = qrObj.viewLocalQ();

    // check that R is square
    const auto m = R.shape(0);
    const auto n = R.shape(1);
    if (m!=n) throw std::runtime_error("The R factor is not square!");

    // -------------------------------------
    // 2. compute SVD of R: R = Ur * Sr * Vr^T
    // each rank performs this since R is replicated
    // note that this returns Vr^T
    const char jobu  = 'S'; // compute only n cols of Ur
    const char jobvt = 'S'; // compute only n rows of of Vr^T
    const auto ldu = m;
    const auto ldv = n;

    if (Ur_.numRows() != m or Ur_.numCols()!=n){
      Ur_.shape(m,n);
    }
    if (S_.shape(0) != n){
      S_.resize({n}, false);
    }
    if (VT_.numRows() != n or VT_.numCols()!=n){
      VT_.shape(n,n);
    }

    int lwork = 5*n;
    std::vector<scalar_t> work(lwork);
    int info = -2;
    lpkObj.GESVD(jobu, jobvt, m, n,
		 R.mutable_data(), m, S_.mutable_data(),
		 Ur_.values(), ldu,
		 VT_.values(), ldv,
		 work.data(), lwork,
		 nullptr, &info);

    if (info!=0) throw std::runtime_error("SVD of R factor failed!");

    // -------------------------------------
    // 3. compute the left-sing vectors U of A as U = Q*Ur
    mat_t teuchosQ(Teuchos::View, Q.mutable_data(),
		   Q.shape(0), Q.shape(0), Q.shape(1));

    if (U_.numRows() != Q.shape(0) or U_.numCols()!=Ur_.numCols()){
      U_.shape(Q.shape(0), Ur_.numCols());
    }

    // make sure we guard when Q.rows is zero
    // which can happen when dealing with sample mesh
    if (Q.shape(0) != 0){
      constexpr auto zero = static_cast<pressiotools::scalar_t>(0);
      constexpr auto one  = static_cast<pressiotools::scalar_t>(1);
      U_.multiply(Teuchos::NO_TRANS, Teuchos::NO_TRANS,
		  one, teuchosQ, Ur_, zero);
    }
  }

  // view singular values (contains all of them since they are replicated)
  pressiotools::py_f_arr viewSingularValues(){
    return S_;
  }

  // view left singular vectors, just the local part
  // because U is row distributed
  pressiotools::py_f_arr viewLocalLeftSingVectorsPy(){
    pressiotools::py_f_arr view({U_.numRows(), U_.numCols()}, U_.values());
    return view;
  }

  // view the right singular vectors transposed,
  // VT_ is the tranpose of V, so the rows of VT_ contain
  // the right singe vectors.
  // VT is replicated, so each rank has a copy of it
  pressiotools::py_f_arr viewRightSingVectorsTransposedPy(){
    pressiotools::py_f_arr view({VT_.numRows(), VT_.numCols()}, VT_.values());
    return view;
  }

  const mat_t & viewLocalLeftSingVectors(){
    return U_;
  }

  const mat_t & viewRightSingVectorsTransposed(){
    return VT_;
  }

private:
  pressiotools::py_f_arr S_;
  mat_t Ur_;
  mat_t U_;
  mat_t VT_;
};

}//end namespace pressiotools
#endif
