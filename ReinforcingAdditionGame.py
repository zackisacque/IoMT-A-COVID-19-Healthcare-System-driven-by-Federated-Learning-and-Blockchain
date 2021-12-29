'''
title           : ReinforcingAdditionGame.py
description     : Towards an Efficient IoMT: Privacy Infrastructure for COVID-19 Pandemic based on Federated Learning and Blockchain Technology
authors         : Omaji Samuel, Akogwu Blessing Omojo, Abdulkarin Musa Onuja, Yunisa Sunday, Prayag Tiwari, Deepak
                  Gupta, Ghulam Hafeez, Adamu Sani Yahaya, Oluwaseun Jumoke Fatoba, and Shahab Shamshirband
date_created    : 20211112
date_modified   : Not Applicable
version         : 0.1
usage           : python FedMedChain.py
                  python FedMedChain.py -p 5000
                  python FedMedChain.py --port 5000
python_version  : 3.7.9
Comments        : A proposed two rounds reinforcing addition game
'''
import random
import math
import numpy as np
from pyglet import input

class ConsensusProtocol:
    '''This is the proposed consensus protocol, which is based on two rounds of reinforcing addition game'''
    def __init__(self):
        ''' Initialization of parameter '''
    
    def TwoRoundsAdditionGame(n):
        '''Step 1: A PRNG is employed to generate the necessary score values for the game. The initial seed is generated
               randomly by the proposed model.
               Step 2: A pseudo-random number is rolled 9 times and on each roll, one player displays the number, which is
                broadcasted to all players in the blockchain network.'''
        # Generate 9 random numbers from 1-9
        val1 = random.randrange(1,n)
        val2 = random.randrange(1,n)
        val3 = random.randrange(1,n)
        val4 = random.randrange(1,n)
        val5 = random.randrange(1,n)
        val6 = random.randrange(1,n)
        val7 = random.randrange(1,n)
        val8 = random.randrange(1,n)
        val9 = random.randrange(1,n)
        matA=[[val1, val2, val3], 
              [val4, val5, val6], 
              [val7, val8, val9]]
        Rol1 = 0 # sum of first row 
        Rol2 = 0 # sum of second row
        Rol3 = 0 # sum of third row
        Col1 = 0 # sum of first column 
        Col2 = 0 # sum of second column
        Col3 = 0 # sum of third column
        Dol1 = 0 # sum of first diagonal 
        Dol2 = 0 # sum of second diagonal
        # iterate through matA
        for i in range(len(matA)):
            Rol1 += matA[0][i]
            Rol2 += matA[1][i]
            Rol3 += matA[2][i]
            Col1 += matA[i][0]
            Col2 += matA[i][1]
            Col3 += matA[i][2]
            for j in range(len(matA)):
                k = 2 - j
                if (i == j):
                    Dol1 += matA[i][j]
                else:
                    Dol2 += matA[j][k]
        ts = Rol1 + Rol2 + Rol3 + Col1 + Col2 + Col3 + Dol1 + Dol2  # total score (ts)
        return ts       

