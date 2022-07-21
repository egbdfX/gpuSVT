# See the LICENSE file at the top-level directory of this distribution.

import ctypes
import numpy as np
try:
    import cupy
except ImportError:
    cupy = None

#import numpy.ctypeslib as ctl
from error import Error
from lib import Lib
from mem import Mem


def cuRandomSVD(matrix_A, rank, oversampling, nIterations):
    A_nRows = matrix_A.shape[0] # matrix A must be stored in column-major format
    A_nCols = matrix_A.shape[1]
    U_nRows = max(1,A_nRows)
    U_nCols = rank # >= max(1,A_nRows)
    V_nRows = rank # >= max(1,A_nCols)
    V_nCols = A_nCols # >= max(1,A_nCols)
    S_nCols = rank # This could be rank
    #print(A_nRows)
    #print(A_nCols)
    #print(U_nCols)
    #print(V_nRows)
    #print(S_nCols)
    
    if type(matrix_A) == np.ndarray:
        matrix_U = np.zeros((U_nRows, U_nCols), dtype=matrix_A.dtype, order='F');
        matrix_V = np.zeros((V_nRows, V_nCols), dtype=matrix_A.dtype, order='C');
        matrix_S = np.zeros(S_nCols, dtype=matrix_A.dtype, order='F');
    elif cupy:
        if type(matrix_A) == cupy.ndarray:
            matrix_U = cupy.zeros((U_nRows, U_nCols), dtype=matrix_A.dtype, order='F');
            matrix_V = cupy.zeros((V_nRows, V_nCols), dtype=matrix_A.dtype, order='C');
            matrix_S = cupy.zeros(S_nCols, dtype=matrix_A.dtype, order='F');
    else:
        raise TypeError("Unknown array type")

    
    mem_S = Mem(matrix_S)
    mem_U = Mem(matrix_U)
    mem_V = Mem(matrix_V)
    mem_A = Mem(matrix_A)
    error_status = Error()
    lib_cuRSVD = Lib.handle().random_svd_python
    lib_cuRSVD.argtypes = [
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        Mem.handle_type(),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int
    ]
    lib_cuRSVD(
        mem_S.handle(),
        mem_U.handle(),
        mem_V.handle(),
        mem_A.handle(),
        rank,
        oversampling,
        nIterations
    )
    error_status.check()
    return(matrix_S, matrix_U, matrix_V)
