from multiprocessing import shared_memory
import numpy as np

def CreateSharedNumpyArray(shape, arrayName='', dataType=np.float64):
    # size of each element in bytes * number of elements
    sharedMemorySize = np.dtype(dataType).itemsize * np.prod(shape)
    sharedMemory = shared_memory.SharedMemory(create=True, size=sharedMemorySize, name=arrayName)

    return arrayName


def ReleaseSharedMemory(arrayNames=['']):
    for iArray in arrayNames:
        sharedMemory = shared_memory.SharedMemory(name=iArray)
        # free and release the shared memory block
        sharedMemory.close()
        sharedMemory.unlink()