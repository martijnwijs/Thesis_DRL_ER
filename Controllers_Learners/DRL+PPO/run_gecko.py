
import multiprocessing
from execute import execute
from itertools import repeat

import os
iterations= 10
global task 
task = 'gait'
global iteration
morphologies = ('gecko6', 'gecko10', 'gecko14')


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
if __name__ == '__main__':    
    for iteration in range(1, (iterations+1)):     
        process_pool = multiprocessing.Pool(processes = 2)                                                        
        process_pool.starmap(execute, zip(morphologies, repeat(iteration)))
