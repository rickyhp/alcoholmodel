# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 00:10:29 2018

@author: rp
"""

import os
import random
from glob import iglob

def renamefiles(pth):
    os.chdir(pth)    
    for name in iglob("*.PNG"):
        try:
            i = random.randint(10000,20000)
            os.rename(name, str(i)+'.png')            
        except OSError:
            print("Caught error for {}".format(name))
            i = random.randint(10000,20000)
            os.rename(name, str(i)+'.png')            
    for name in iglob("*.JPG"):
        try:
            i = random.randint(20000,30000)
            os.rename(name, str(i)+'.jpg')            
        except OSError as e:
            print("Caught error for {}".format(name), ' : ' , e)
            i = random.randint(20000,30000)
            os.rename(name, str(i)+'.jpg')
            
renamefiles('F:\\CODE\\Alcohol\\dataset\\train\\others')