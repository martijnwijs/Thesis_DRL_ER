    
import os
def execute(morphologie, iteration):      

    os.system(f"python optimize.py --from_checkpoint {morphologie} rotation {iteration}")