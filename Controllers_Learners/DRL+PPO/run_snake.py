
import multiprocessing
from execute import execute
from itertools import repeat

import os
iterations= 10
global task 
task = 'gait'
global iteration
morphologies = ('snake6', 'snake10', 'snake14')


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
if __name__ == '__main__':    
    # This block of code enables us to call the script from command line. 
                                                                                  
 #   def execute(morphologie, iteration):                                                             
 #       os.system(f"python optimize.py --from_checkpoint {morphologie} {task} {iteration}")                                       

    for iteration in range(1, (iterations+1)):     
        #iterations = [iteration, iteration, iteration, iteration, iteration, iteration, iteration, iteration, iteration, iteration, iteration, iteration]
        
        process_pool = multiprocessing.Pool(processes = 2)                                                        
        process_pool.starmap(execute, zip(morphologies, repeat(iteration)))
