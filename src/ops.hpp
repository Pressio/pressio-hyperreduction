
#ifndef PRESSIOTOOLS_OPS_HPP_
#define PRESSIOTOOLS_OPS_HPP_

#include "Epetra_MpiComm.h"
#include "Epetra_MultiVector.h"
#include "Epetra_Vector.h"
#include <Epetra_Import.h>
#include <Epetra_Map.h>
#include <Epetra_LocalMap.h>

namespace pressiotools{ namespace ops{

// C = beta * C + alpha*op(A)*op(B)
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
	int rc = epA(i)->Dot( *(epB(j)), &tmp );
	if (rc!=0){
	  throw std::runtime_error("Error computing epetra dot");
	}
	C(i,j) += alpha*tmp;
      }
    }
  }
  else if (oA == 'N' and oB=='T')
  {
    using mv_t	= Epetra_MultiVector;
    Epetra_MpiComm commE(A.communicator());

    // compute C = A B^T where
    // A,C are distriubted and B is imported to be locally replicated

    // epetra wrapper for A
    Epetra_Map mapA(A.extentGlobal(0), A.extentLocal(0), 0, commE);
    scalar_t * ptrA = A.data().mutable_data();
    mv_t epA(Epetra_DataAccess::View, mapA, ptrA, A.extentLocal(0), A.extentLocal(1));

    // epetra wrapper for B
    Epetra_Map mapB(B.extentGlobal(0), B.extentLocal(0), 0, commE);
    scalar_t * ptrB = B.data().mutable_data();
    mv_t epB(Epetra_DataAccess::View, mapB, ptrB, B.extentLocal(0), B.extentLocal(1));
    // convert it to replicated matrix
    // we have to do this otherwise multiply below does not work
    Epetra_LocalMap locMap(B.extentGlobal(0), 0, commE);
    Epetra_Import importer(locMap, mapB);
    mv_t epBLocRep(locMap, B.extentGlobal(1));
    int rc = epBLocRep.Import(epB, importer, Insert);
    if (rc!=0){
      throw std::runtime_error("Error doing redistribution");
    }

    scalar_t * ptrC = C.mutable_data();
    auto mapC = mapA;
    mv_t epC(Epetra_DataAccess::View, mapC, ptrC, C.shape(0), C.shape(1));
    int rc2 = epC.Multiply('N', 'T', 1., epA, epBLocRep, 0.);
    if (rc2!=0){
      throw std::runtime_error("Error computing C = A B^T");
    }
  }
  else{
    throw std::runtime_error("Case not yet implmented");
  }
}

// C = alpha*A^T*A
void selfTransposeSelf(const pressiotools::scalar_t alpha,
		       pressiotools::MultiVector & A,
		       pressiotools::py_f_arr C)
{
  assert((std::size_t)C.extent(0) == (std::size_t)A.extentLocal(1));
  assert((std::size_t)C.extent(1) == (std::size_t)A.extentLocal(1));

  auto & comm = A.communicator();
  const int globNumRows = A.extentGlobal(0);

  Epetra_MpiComm commE(comm);
  Epetra_Map map(globNumRows, A.extentLocal(0), 0, commE);

  scalar_t * ptrA = A.data().mutable_data();
  using mv_t	= Epetra_MultiVector;
  mv_t epA(Epetra_DataAccess::View, map, ptrA, A.extentLocal(0), A.extentLocal(1));

  // C = beta* C + alpha*A^T*A
  // beta is zero for now
  auto beta = static_cast<pressiotools::scalar_t>(0);

  auto tmp = static_cast<pressiotools::scalar_t>(0);
  for (int i=0; i<A.extentLocal(1); i++)
  {
    C(i,i) = beta*C(i,i);
    epA(i)->Dot( *(epA(i)), &tmp );
    C(i,i) += alpha*tmp;

    for (int j=i+1; j<A.extentLocal(1); j++)
    {
      C(i,j) = beta*C(i,j);
      C(j,i) = beta*C(j,i);
      epA(i)->Dot( *(epA(j)), &tmp );
      C(i,j) += alpha*tmp;
      C(j,i) += alpha*tmp;
    }
  }
}

}}//end namespace pressiotools::ops
#endif
