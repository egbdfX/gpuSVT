from cuRandomSVD import *

A_nRows = 5;
A_nCols = 5;
nIterations = 2;
rank = 2;
p = 2;
U_nCols = A_nRows; # >= max(1,A_nRows)
V_nRows = A_nCols; # >= max(1,A_nCols)
S_nCols = min(A_nRows,A_nCols) # This could be rank


A_size_bytes = A_nRows*A_nCols;
U_size_bytes = A_nRows*U_nCols;
V_size_bytes = V_nRows*A_nCols;
S_size_bytes = S_nCols;

matrix_A = np.array([[0.764207, 0.0369729, 0.420183, 0.503398, 0.583257] , [0.614115, 0.859624, 0.392045, 0.92975, 0.115899] , [0.817242, 0.675841, 0.126579, 0.21214, 0.398311] , [0.420409, 0.455947, 0.902506, 0.639625, 0.214927] , [0.0344609, 0.0207483, 0.230762, 0.581246, 0.00540355]], dtype=np.float64, order='F')

matrix_A_cupy = cupy.asarray(matrix_A, dtype=matrix_A.dtype, order='F');  

print("Input matrix:\n")
print(matrix_A)

print("----------------------------------------------")

S, U, V = cuRandomSVD(matrix_A_cupy, rank, p, nIterations);

print("GPU computed values:")
print('S:\n',S)
print('U:\n',U)
print('V:\n',V)
print("----------------------------------------------")

Upy, Dpy, VTpy = np.linalg.svd(matrix_A)

print("CPU computed values:")
print('S:\n',Dpy)
print('U:\n',Upy)
print('V:\n',VTpy)
