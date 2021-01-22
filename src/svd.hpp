
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

    // 1. compute A = QR
    qr_.computeThinOutOfPlace(A);

    // view factors (note that these have view semantics)
    // R is replicated on each rank, square, smallish matrix
    auto R = qr_.viewR();
    auto Q = qr_.viewLocalQ();

    // check that R is square
    const auto m = R.shape(0);
    const auto n = R.shape(1);
    if (m!=n) throw std::runtime_error("The R factor is not square!");
    // -------------------------------------

    // 2. compute SVD of R: R = Ur * Sr * Vr
    const char jobu = 'S'; // compute only n vectors of U
    const char jobv = 'S'; // compute only n rows of of V^T
    const auto ldu = m;
    const auto ldv = n;

    if (Ur_.numRows() != m or Ur_.numCols()!=n) Ur_.shape(m,n);
    if (S_.shape(0) != n) S_.resize({n}, false);
    if (VT_.numRows() != n or VT_.numCols()!=n) VT_.shape(n,n);

    int lwork = 5*n;
    std::vector<scalar_t> work(lwork);
    int info = -2;
    lpk_.GESVD(jobu, jobv, m, n,
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

    if (U_.numRows() != Q.shape(0) or U_.numCols()!=Ur_.numCols())
      U_.shape(Q.shape(0), Ur_.numCols());

    constexpr auto zero = static_cast<pressiotools::scalar_t>(0);
    constexpr auto one  = static_cast<pressiotools::scalar_t>(1);
    U_.multiply(Teuchos::NO_TRANS, Teuchos::NO_TRANS,
		one, teuchosQ, Ur_, zero);
  }

  pressiotools::py_f_arr viewS(){ return S_; }

  pressiotools::py_f_arr viewU(){
    pressiotools::py_f_arr view({U_.numRows(), U_.numCols()}, U_.values());
    return view;
  }

  pressiotools::py_f_arr viewVT(){
    pressiotools::py_f_arr view({VT_.numRows(), VT_.numCols()}, VT_.values());
    return view;
  }

  const mat_t & viewNativeU(){ return U_; }
  const mat_t & viewNativeVT(){ return VT_; }

private:
  Tsqr qr_;
  Teuchos::LAPACK<int, pressiotools::scalar_t> lpk_;
  pressiotools::py_f_arr S_;

  mat_t Ur_;
  mat_t U_;
  mat_t VT_;
};

}//end namespace pressiotools
#endif
