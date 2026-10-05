import sys
from os import getcwd, makedirs

sys.path.append(getcwd())

from time import perf_counter
import numpy as np
from utilities import CustomMatmul
from prettytable import PrettyTable

#######################################################################
#  Versions are: 'numpy', 'cudaSimple', 'cudaTranspose', 'cudaShmem'  #
#                                                                     #
#         Fill in the list with the versions you've implemented       #
#                                                                     #
#!-!-!-!-!-!-!-!-!-!-!-!-Do not remove numpy-!-!-!-!-!-!-!-!-!-!-!-!-!#
#######################################################################

if __name__ == '__main__':
    
    versions = ['numpy', 'cudaSimple', 'cudaTranspose', 'cudaShmem'] 

    # Dimensions for the square matrices to be tested
    squareDimensions = [1024, 2048, 4096, 6144, 8192]

    outTableHeader = [
                    'Version',
                    'Dimensions',
                    'isValid',
                    'Total Time (ms)',
                    'Execution Time (ms)',
                    'Transfers Time (ms)',
                    'speedup to numpy'
                    ]

    resultsList = [outTableHeader]        
    outTable = PrettyTable(outTableHeader)

    csvHeader = ['Version', 'SquareMatrixDimension', 'TotalTime', 'ExecutionTime', 'TransfersTime', 'speedupToNumpy']
    csvTable = PrettyTable(csvHeader)

    bigTic = perf_counter()

    for iSquareMatrixDim in squareDimensions:
        # Create random matrices A and B only when needed to save RAM!
        A = np.random.randn(iSquareMatrixDim, iSquareMatrixDim).astype('float64')
        B = np.random.randn(iSquareMatrixDim, iSquareMatrixDim).astype('float64')

        numpyTest = None
        numpyTestTime = None

        for iVersion in versions:
            
            if iVersion == "numpy":
                # Numpy test (no warmup needed, CustomMatmul for numpy returns out, execTime)
                numpyTest, numpyTestTime = CustomMatmul(A, B, mode="numpy")
                totalTime = numpyTestTime
                transfersTime = 0.0
            else:
                # Warmup for CUDA (only 1 warmup needed to compile JIT and init CUDA context)
                # We skip warming up numpy because it is unnecessary.
                CustomMatmul(A, B, mode=iVersion)

                tic = perf_counter()
                iVersionTest, transfersTime = CustomMatmul(A, B, mode=iVersion)
                toc = perf_counter()
                totalTime = toc - tic
            
            # Check validity per dimension/version instead of globally across all dims
            if iVersion == "numpy":
                isValid = True
            else:
                isValid = np.allclose(numpyTest, iVersionTest)

            speedup = numpyTestTime / totalTime

            iRow = [
                    iVersion,
                    f'({iSquareMatrixDim}x{iSquareMatrixDim}) • ({iSquareMatrixDim}x{iSquareMatrixDim})', 
                    isValid,
                    np.round(1000*totalTime, 2),  
                    np.round(1000*(totalTime-transfersTime), 2),
                    np.round(1000*transfersTime, 2),
                    np.round(speedup, 2),  
                ]

            outTable.add_row(iRow)
            csvTable.add_row([
                    iVersion,
                    iSquareMatrixDim,
                    np.round(1000*totalTime, 2),  
                    np.round(1000*(totalTime-transfersTime), 2),
                    np.round(1000*transfersTime, 2),
                    np.round(speedup, 2)
            ])
            
        outTable.add_row(['', '', '', '', '', '', ''])

    bigToc = perf_counter()

    makedirs('results', exist_ok=True)
    with open(f'results/CUDAMatMulTests.out', 'w+') as f:
        print(outTable, file=f)
        print("Total Time:", np.round((bigToc-bigTic)/60, 2), "mins", file=f)

    makedirs('plotFiles', exist_ok=True)
    with open(f'plotFiles/CUDAMatMulTests.csv', 'w+') as f:
        f.write(csvTable.get_csv_string())
