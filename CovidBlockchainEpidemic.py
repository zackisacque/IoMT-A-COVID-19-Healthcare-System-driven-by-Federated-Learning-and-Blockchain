'''
title           : CovidBlockchainEpidemic.py
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
Comments        : We formulate the proposed mathematical model (SESIAISQEQHR) for the COVID-19 pandemic. Eight compartments are considered in the proposed 
                  mathematical model, which are susceptible ($S$), exposure ($E$), susceptible infectious ($I_{s}$), asymptomatic infectious 
                  ($I_{a}$), susceptible quarantine ($S_{q}$), expose quarantine ($E_{q}$), hospitalized ($H$), recovery ($R$) and blockchain
                  impact ($B$). 
'''
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Iterable, Any
from statsmodels.tools.eval_measures import aic_sigma
from scipy.special import logsumexp
from scipy import float64

model_types = {
    'SESIAISQEQHR': {
        'variables': {'S': 'Susceptible', 'E': 'Exposed', 'SI': ' Symptomatic Infectious', 'AI': 'Asymptomatic Infectious', 'SQ': 'Susceptible Quarantine',
                      'EQ': 'Exposed Quarantine', 'H': 'Hospitalized', 'R': 'Recovery', 'B': 'Blockchain Impact'},
        'parameters': {'phiB': r'$\phi_{B}$', 'Rq': r'$R_{q}$', 'PT': r' $P_{T}$',
                       'sigma': r' $\sigma$', 'lambdaB': r' $\lambda_{B}$', 'Ca': r' $C_{a}$', 'Cs': r'$C_{s}$',
                       'phis': r'$\phi_{s}$', 'taus': r'$\tau_{s}$', 'taua': r'$\tau_{a}$', 'tauh': r'$\tau_{h}$',
                       'deltas': r'$\delta_{s}$', 'deltaa': r'$\delta_{a}$', 'deltaq': r'$\delta_{q}$', 'Cm': ' $C_{m}$',
                       'deltah': r'$\delta_{h}$', 'muB': r'$\mu_{B}$', 'etaB': ' $\eta_{B}$', '(1-r)': '$(1-r)$', '(1-C_{m})': '$(1-C_{m})$',
                       '(1-C_{q})': '$(1-C_{q})$', 'Bc': '$B_{c}$', 'Ba': '$B_{a}$', 'Hr': r'$H_{r}$'
                       }
        }
}

class SESIAISQEQHR():
    ''' This class provides the impact of public awareness induced by blockchain technology'''
    def __init__(self):
        """
        Difference equation based model (discrete time)
        """
        super().__init__()

    def run(self, *args):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        res = self.run(*args)
        #self.traces.update(res)
        # return res
    def __init__(self):
        super().__init__()
        self.model_type = 'SESIAISQEQHR'
        self.state_variables = {'S': 'Susceptible', 'E': 'Exposed', 'SI': ' Symptomatic Infectious', 'AI': 'Asymptomatic Infectious', 'SQ': 'Susceptible Quarantine',
                      'EQ': 'Exposed Quarantine', 'H': 'Hospitalized', 'R': 'Recovery', 'B': 'Blockchain Impact'}
        self.parameters = {'phiB': r'$\phi_{B}$', 'Rq': r'$R_{q}$', 'PT': r' $P_{T}$',
                       'sigma': r' $\sigma$', 'lambdaB': r' $\lambda_{B}$', 'Ca': r' $C_{a}$', 'Cs': r'$C_{s}$',
                       'phis': r'$\phi_{s}$', 'taus': r'$\tau_{s}$', 'taua': r'$\tau_{a}$', 'tauh': r'$\tau_{h}$',
                       'deltas': r'$\delta_{s}$', 'deltaa': r'$\delta_{a}$', 'deltaq': r'$\delta_{q}$', 'Cm': ' $C_{m}$',
                       'deltah': r'$\delta_{h}$', 'muB': r'$\mu_{B}$', 'etaB': ' $\eta_{B}$', 'r': '$(1-r)$', 'Cm1': '$(1-C_{m})$',
                       'Cq1': '$(1-C_{q})$', 'Bc': '$B_{c}$', 'Ba': '$B_{a}$', 'Hr': r'$H_{r}$'
                       }
        self.run = self.model

    def model(self, inits: list, trange: list, totlpop: int, params: dict) -> dict:
        """
        calculates the model SESIAISQEQHR, and return its values (no demographics)
        - inits = (S,E,SI,AI,SQ,EQ,H,R)
        - B = blockchain impact
        :param trange:
        :param params:
        :param inits: tuple with initial conditions
        :param simstep: step of the simulation
        :param totlpop: total population
        :S(0) = Susceptible individuals 
        :SQ(0)= Susceptible quarantined individuals  
        :E(0) = Exposed Individuals 
        :EQ(0) = Exposed quarantined individuals
        :SI(0) = Infected individuals with symptomatic 
        :AI(0) = Infected individuals with asymptomatic 
        :H(0) = Hospitalized individuals 
        :R(0) = Recovered individuals 
        :B = Rate of blockchain impact 
        :phiB = Weight of blockchain effect on contact rate 
        :Rq = Releasing rate of quarantined individuals per day 
        :PT = Transmission probability of effective infected individuals per contact
        :sigma = Progression from exposed compartment to infected compartment per day 
        :lambdaB = Quarantined rate induced by awareness driven by blockchain technology 
        :Ca = Effective contact rates for asymptomatic individual per day 
        :Cs = Effective contact rates for symptomatic individual per day 
        :phis = Hospitalized rate of infected individual per day 
        :taus = Recovery rates of individuals in SI compartment per day 
        :taua = Recovery rates of individuals in AI compartment per day  
        :tauh = Recovery rates of individuals in  H compartment per day
        :deltas = Hospitalized rates of individuals in SI 
        :deltaa = Hospitalized rates of individuals in AI 
        :deltaq = Hospitalized rates of individuals in EQ
        :Cm = Proportion of individuals exposed to COVID-19 virus during contact tracing
        :deltah = Disease-induced death rate 
        :muB = Reduction rate of public awareness due to limitations of blockchain technology
        :etaB = Response intensity of blockchain technology on newly confirmed cases
        :r = Fraction of exposed individuals who show symptoms at the end of the incubation period
        :Cm1 = Proportion of individuals missed during contact tracing 
        :Cq1 = Proportion of individuals willing to stay quarantined 
        :Bc = Rate of block creation by the leader per minutes 
        :Ba = Number of awareness driven by blockchain technology 
        :Hr = Relative hash power of CDC for block creation per second
        :return:
        """
        # SQ: np.ndarray = np.zeros(trange[1] - trange[0])
        #np.seterr('raise')
        # SQ: np.ndarray = np.zeros(trange[1] - trange[0])
        #np.seterr('raise')
        S: np.array(dtype=np.float64) = np.ones(trange[1] - trange[0])
        SQ: np.array(dtype=np.float64) = np.ones(trange[1] - trange[0])
        E: np.array(dtype=np.float64) = np.ones(trange[1] - trange[0])
        EQ: np.array(dtype=np.float64) = np.ones(trange[1] - trange[0])
        SI: np.array(dtype=np.float64) = np.ones(trange[1] - trange[0])
        AI: np.array(dtype=np.float64) = np.ones(trange[1] - trange[0])
        H: np.array(dtype=np.float64) = np.ones(trange[1] - trange[0])
        R: np.array(dtype=np.float64) = np.ones(trange[1] - trange[0])
        B: np.array(dtype=np.float64) = np.ones(trange[1] - trange[0])
        #Lpos1: np.ndarray = np.zeros(trange[1] - trange[0])
        #Lpos2: np.ndarray = np.zeros(trange[1] - trange[0])
        tspan = np.arange(*trange)

        S[0], SQ[0], E[0], EQ[0], SI[0], AI[0], H[0], R[0], B[0] = inits
        N = totlpop
        
        phiB = params['phiB'];
        Rq = params['Rq']; 
        PT = params['PT'];
        sigma = params['sigma'];
        lambdaB = params['lambdaB'];
        Ca = params['Ca'];
        Cs = params['Cs'];
        phis = params['phis'];
        taus = params['taus'];
        taua = params['taua'];
        tauh = params['tauh'];
        deltas = params['deltas']; 
        deltaa = params['deltaa']; 
        deltaq = params['deltaq'];
        Cm = params['Cm'];
        deltah = params['deltah'];
        muB = params['muB'];
        etaB = params['etaB'];
        r = params['r'];
        Cm1 = params['Cm1'];
        Cq1 = params['Cq1'];
        Bc = params['Bc'];
        Ba = params['Ba'];
        Hr = params['Hr'];
        Lpos1 = S[0] * SI[0] / N  # Number of new cases for symptomatic
        Lpos2 = S[0] * AI[0] / N  # Number of new cases for asymptomatic
        for i in tspan[:-1]:
            #print('lpos:',Lpos1)
            S[i + 1] = np.isnan((Cs * phiB * Cq1 * Lpos1 * S[i])-(Ca * phiB * Cq1 * Lpos1 * S[i])-(lambdaB * S[i] * B[i] + Rq * SQ[i]))
            #print('S:', S)
            E[i + 1] =  np.isnan((Cs * phiB * Cm1 * PT * Lpos2) - (Ca * phiB * Cm1 * PT * Lpos2 * S[i])- (sigma + lambdaB * B[i]) * E[i])
            SI[i + 1] = np.isnan((r *  sigma * E[i]) - (phis + taus + deltas) * SI[i])
            AI[i + 1] =  np.isnan((r * sigma * E[i]) - (taua * AI[i]))
            SQ[i + 1] =  np.isnan((Cs * phiB * Cm1 * PT) + (lambdaB * S[i] * B[i]) - (Rq * SQ[i]))
            EQ[i + 1] =  np.isnan((Cs * phiB * PT * Cm) + (lambdaB * E[i] * B[i]) -(deltaq * EQ[i]))
            H[i + 1] =  np.isnan((phis * SI[i]) + (deltaq * EQ[i]) -(tauh * deltah) * H[i])
            R[i + 1] =  np.isnan((taus * SI[i]) + (taua * AI[i]) + (tauh * AI[i]))
            B[i + 1] =  np.isnan(etaB * ((deltas * SI[i]) + (deltaa * AI[i]) + (deltaq * EQ[i])) - (muB * B[i]))
          
        return {'time': tspan, 'S': S, 'E': E, 'SI': SI, 'AI': AI,'SQ': SQ, 'EQ': EQ, 'H': H, 'R': R, 'B': B}
    
#if __name__ == "__main__":
    #modelSESIAISQEQHR = SESIAISQEQHR()
    #modelSESIAISQEQHR([150, 100, 100, 50, 20, 1, 1, 2, 16],
             #[0,50],
             #100000,
             #{'phiB': 0.08 ,
              # 'Rq': 0.7, 
               #'PT': 0.8,
               #'sigma': 0.2,
               #'lambdaB': 0.5,
               # 'Ca': 0.2, 
                #'Cs': 0.1,
                #'phis': 0.3, 
                #'taus': 0.3,
                #'taua': 0.201,
                #'tauh': 0.130,
                #'deltas': 0.11,
                #'deltaa': 0.102,
                #'deltaq': 0.2,
                #'Cm': 0.2,
                #'deltah': 0.003,
                #'muB': 0.3,
                #'etaB': 0.735,
                #'r': 0.13,
                #'Cm1': 0.5,
                #'Cq1': 0.3,
                #'Bc': 10,
                #'Ba': 100,
                #'Hr': 1414
            #})