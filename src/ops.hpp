
#ifndef PRESSIOTOOLS_OPS_HPP_
#define PRESSIOTOOLS_OPS_HPP_

#include "Epetra_MpiComm.h"
#include "Epetra_MultiVector.h"
#include "Epetra_Vector.h"

namespace pressiotools{ namespace ops{

void product(const char oA,
	     const char  oB,
	     const pressiotools::scalar_t alpha,
	     pressiotools::MultiVector & A,
	     pressiotools::MultiVector & B,
	     const pressiotools::scalar_t beta,
	     pressiotools::py_f_arr C)
{
  if (oA == 'T' and oB=='N')
  {
    assert((std::size_t)A.extentGlobal(0) == (std::size_t)B.extentGlobal(0));
    assert((std::size_t)A.extentLocal(0) == (std::size_t)B.extentLocal(0));
    assert((std::size_t)C.extent(0) == (std::size_t)A.extentLocal(1));
    assert((std::size_t)C.extent(1) == (std::size_t)B.extentLocal(1));

    auto & comm = A.communicator();
    const int globNumRows = A.extentGlobal(0);

    Epetra_MpiComm commE(comm);
    Epetra_Map map(globNumRows, A.extentLocal(0), 0, commE);

    scalar_t * ptrA = A.data().mutable_data();
    scalar_t * ptrB = B.data().mutable_data();
    using mv_t	= Epetra_MultiVector;
    mv_t epA(Epetra_DataAccess::View, map, ptrA, A.extentLocal(0), A.extentLocal(1));
    mv_t epB(Epetra_DataAccess::View, map, ptrB, B.extentLocal(0), B.extentLocal(1));

    auto tmp = static_cast<pressiotools::scalar_t>(0);
    // compute dot between every column of A with every col of B
    for (int i=0; i<A.extentLocal(1); i++)
    {
      for (int j=0; j<B.extentLocal(1); j++)
      {
	C(i,j) = beta*C(i,j);
	epA(i)->Dot( *(epB(j)), &tmp );
	C(i,j) += alpha*tmp;
      }
    }
  }
  else{
    throw std::runtime_error("Case not yet implmented");
  }
}

}}//end namespace pressiotools::ops
#endif
