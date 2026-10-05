import sys
from os import getcwd, makedirs

sys.path.append(getcwd())

from time import perf_counter
import numpy as np
from utilities import CustomMatmul
from prettytable import PrettyTable
from multiprocessing import shared_memory
from Multiprocessing.tools import ReleaseSharedMemory, CreateSharedNumpyArray

if __name__ == '__main__':

    numberOfProcesses = [1, 5, 10, 20, 40]

    # Dimensions for the square matrices to be tested
    squareDimensions = [100, 200, 400, 600]

    outTableHeader = ['Number of Processes', 'Dimensions', 'Multiprocessing Time', 'speedup to serial', 'Multiprocessing isValid']
    resultsList = [outTableHeader]        
    outTable = PrettyTable(outTableHeader)

    csvHeader = ['NumberOfProcesses', 'SquareMatrixDimension', 'ExecutionTime', 'speedupToSerial']
    csvTable = PrettyTable(csvHeader)
    
    bigTic = perf_counter()

    for iSquareMatrixDim in squareDimensions:
        # Create random matrices A and B
        A_temp = np.random.randn(iSquareMatrixDim, iSquareMatrixDim).astype('float64')
        B_temp = np.random.randn(iSquareMatrixDim, iSquareMatrixDim).astype('float64')

        # Numpy Test
        numpyTest, numpyTestTime = CustomMatmul(A_temp, B_temp, mode="numpy")

        # Multiprocessing Tests

        # Create shared memory for A and B
        outArrayShape = (A_temp.shape[0], A_temp.shape[1])
        CreateSharedNumpyArray(outArrayShape, arrayName='A')
        shmA = shared_memory.SharedMemory(name='A')
        A = np.ndarray(outArrayShape, dtype=np.float64, buffer=shmA.buf)
        A[:] = A_temp[:,:]

        outArrayShape = (B_temp.shape[0], B_temp.shape[1])
        CreateSharedNumpyArray(outArrayShape, arrayName='B')
        shmB = shared_memory.SharedMemory(name='B')
        B = np.ndarray(outArrayShape, dtype=np.float64, buffer=shmB.buf)
        B[:] = B_temp[:,:]

        serialTime = 1

        # Test with different number of processes
        for iNumProcesses in numberOfProcesses:
            multiprocessingTest, multiprocessingTestTime = CustomMatmul(A, B, mode="multiprocessing", numberOfProcesses=iNumProcesses)
            if iNumProcesses==1:
                serialTime = multiprocessingTestTime

            iRow = [
                    iNumProcesses,
                    f'({iSquareMatrixDim}x{iSquareMatrixDim}) • ({iSquareMatrixDim}x{iSquareMatrixDim})',
                    np.round(multiprocessingTestTime, 3),
                    np.round(serialTime/multiprocessingTestTime, 3),
                    np.allclose(numpyTest, multiprocessingTest)
                    ]
            
            outTable.add_row(iRow)
            csvTable.add_row([
                    iNumProcesses,
                    iSquareMatrixDim,
                    np.round(multiprocessingTestTime, 3),
                    np.round(serialTime/multiprocessingTestTime, 3),
            ])
        outTable.add_row(['', '', '', '', ''])
        ReleaseSharedMemory(['A', 'B'])

    bigToc = perf_counter()

    makedirs('results', exist_ok=True)
    with open(f'results/MultiprocMatMulTests.out', 'w+') as f:
        print(outTable, file=f)
        print("Total Time:", np.round((bigToc-bigTic)/60, 2), "mins", file=f)

        f.close()

    makedirs('plotFiles', exist_ok=True)
    with open(f'plotFiles/MultiprocMatMulTests.csv', 'w+') as f:
        f.write(csvTable.get_csv_string())
        f.close()
