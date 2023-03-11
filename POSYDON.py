#!/usr/bin/env python
# coding: utf-8

#author: Petros Papadopoulos

#%%

import gurobipy as gp
import numpy as np
import pandas as pd
import time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

#%%

def POSYDON(norm_power,
            norm_powerL,
            el_price,
            el_priceL,
            op_time,
            op_timeL,
            RUL_nom,
            RUL_LG,
            RUL_LG_LTH,
            access_LTH,
            cycle_time,
            rho=None,
            curtailment=0,
            scalars=None,
            STH_sol=None,
            return_all_vars=False,
            rel_gap=0.001,
            CMS=False,
            TBS=False,
            NOYAW=False
           ):
    '''
    Mathematical optimization model that uses the Gurobi solver. 
    
    Inputs:
    - norm_power: np.array of shape (24,N_t,N_J,N_S) -> hourly normalized power scenarios of the wind turbines in the STH
    - norm_powerL: np.array of shape (LTH_dim,N_t,N_J,S) -> daily normalized power scenarios of the wind turbines in the LTH
    - el_price: np.array of shape (24,N_S) -> hourly electricity price scenarios in the STH [$/MWh]
    - el_priceL: np.array of shape (LTH_dim,N_S) -> daily electricity price scenarios in the LTH [$/MWh]
    - op_time: np.array of shape (24,N_t,N_S) -> hourly operation time scenarios for turbine maintenance [hours]
    - op_timeL: np.array of shape (LTH_dim,N_t,N_S) -> daily operation time scenarios for turbine maintenance [hours]
    - RUL_nom: np.array of shape (N_t,N_S) -> maximum residual life estimate scenarios for each wind turbine [days]
    - RUL_LG: np.array of shape (24, N_t, N_J, N_S) -> hourly damage equivalent load [days]
    - RUL_LG_LTH: np.array of shape (LTH_dim, N_t, N_J, N_S) -> daily damage equivalent load [days]
    - access_LTH: np.array of shape (LTH_dim,N_t,N_S) -> daily operation time scenarios for turbine maintenance [hours]
    - cycle_time: np.array of shape (N_t) -> time since last repair cycle [days]
    - rho: (optional) np.array of shape (N_t,) -> binary parameter denoting whether maintenance was initiated in the past.
        By default no maintenance was initiated in the past.. rho = np.zeros(N_t)
    - curtailment: (optional) np.array of shape (24,N_S) -> hourly power curtailment (normalized). Default value is zero. 
    - scalars: (optional) tuple or list containing scalar parameter values with the following order:
        Kappa (default: 4000) Cost of each PM task [$]
        Fi (default: 10000) Cost of each CM task [$]
        Psi (default: 250) Maintenance crew hourly cost [$/h]
        Omega (default: 2500) Vessel daily rental cost [$/d]
        Q (default: 125) Maintenance crew overtime cost [$/h]
        R (default: 12) Wind turbine rated power output [MW]
        W (default: 8) Number of workhours with standard payment [hours]
        B (default: 2) Number of maintenance crews
        H (default: 8) Max overtime hours in total [hours]
        tR (default: 5) Time of first sunlight
        tD (default: 21) Time of last sunlight
    - STH_sol: (optional) -> a list with two elements: the day-ahead maintenance and yaw misalignment schedules saved as np.arrays
    - return_all_vars: (optional) bool -> Returns the whole optimization model instead of selected processed outputs. 
        Default value is False to minimize memory usage.
    - rel_gap: (optional) float -> Sets the relative optimality gap of the optimizer.
        
    Returns:
    - output: A dictionary (only if return_all_vars is set to False) with the following keys: 
        'STH_PM_sched': pd.DataFrame of shape (STH_dim,N_t) with the PM schedule of the STH.
        'STH_CM_sched': pd.DataFrame of shape (STH_dim,N_t) with the CM schedule of the STH. 
        'LTH_PM_sched': np.array of shape (STH_dim,N_t,N_S) with the stochastic PM schedule of the LTH. 
        'LTH_CM_sched': np.array of shape (STH_dim,N_t,N_S) with the stochastic CM schedule of the LTH.
        'expected_STH_cost': The expected cost of the STH.
        'expected_LTH_cost': The expected cost of the LTH.
        'total_expected_cost': Total expected cost, which is the sum of the STH and LTH cost, plus the expected cost
            incured from unfinished maintenance tasks in the STH.
        'remaining_maint_hrs': np.array of shape (N_t, N_S) with the number of maintenance hours remaining for tasks that
            where initiated in the STH but not completed in the same day.
        'model': None, or a dict containing  all the variable values of the optimization model (only if 
            return_all_vars is set to True)
    '''
    
    ## Construct the sets ##
    tF, N_t, N_J, N_S  = norm_power.shape
    N_D = norm_powerL.shape[0]
    
    I = ['wt'+str(i) for i in range(1,N_t+1)]
    T = ['t'+str(t) for t in range(tF)]
    D = ['d'+str(d) for d in range(1,N_D+1)]
    S = ['s'+str(s) for s in range(1,1+N_S)]
    J = ['j'+str(j) for j in range(1,1+N_J)]
    
    
    ## Specify default parameters ##
    rho = pd.Series(np.zeros(N_t), index = I) if rho is None else pd.Series(rho, index=I)
    
    #Cu = pd.DataFrame(np.ones((tF,N_S))-curtailment, columns=S, index=T).T
    
    # Scalar parameters and costs
    if scalars is None:
        Kappa = 4000      #($/PM task) Cost of each PM task
        Fi = 10000        #($/CM task) Cost of each CM task
        Psi = 250         #($/h) Maintenance crew hourly cost
        Omega = 2500      #($/d) Vessel daily rental cost
        Q = 125           #($/h) Maintenance crew overtime cost
        R = 12            #(MW) Wind turbine rated power output
        W = 8             #(ts) Number of workts with standard payment
        B = 2             #(-) Number of maintenance crews
        if N_t>=40: B = 4
        H = 8
        tR = 5   
        tD = 21
        C_lambda = 30 #($/day) before: 2000/360 + 1500/600 #lifecycle reduction cost
        N_theta = N_D+1
        if N_S==1: N_theta *= 2 #if a deterministc version is used, increase N_tyheta to make more risk averse
        N_TBS = 70 #(days) the TBS maintenance visit intervals
    else:
        Kappa,Fi,Psi,Omega,Q,R,W,B,H,tR,tD,C_lambda,N_theta,N_TBS = scalars
        
        
    
    
    if TBS: #In TBS, if the maintenance period has been reached or unexpected failure, theta=1
        assert len(S)==1, 'TBS is a deterministic benchmark.'
        theta_i = (N_TBS-cycle_time<=0) | (RUL_nom[:,0]<1)
        N_theta=0
    
    
    
    
    #calculate setaL for the nominal loading case
    zeta_0 = np.zeros((N_D, N_t, N_S))
    for i in range(N_t):
        zeta_0[:,i,:] = np.repeat(np.arange(2,N_D+2).reshape(-1,1),N_S, 1)<=RUL_nom[i,:].reshape(1,-1)    
    
    
    
    ## Convert stochastic parameters to appropriate formats (dict) ##
    Pi = {t:{s: el_price[ti,si] for si, s in enumerate(S)} for ti, t in enumerate(T) }
    PiL = {d:{s: el_price[di,si] for si, s in enumerate(S)} for di, d in enumerate(D) }
    norm_power2 = norm_power.copy()
    norm_power2[norm_power2<=1e-4]=1e-4
    f = {t: {i: {j: {s: norm_power2[ti,ii,ji,si] for si, s in enumerate(S)}
                 for ji, j in enumerate(J)} for ii, i in enumerate(I)} 
                 for ti, t in enumerate(T)}
    fL = {d: {i: {j: {s: norm_powerL[ti,ii,ji,si] for si, s in enumerate(S)}
                 for ji, j in enumerate(J)} for ii, i in enumerate(I)} 
                 for ti, d in enumerate(D)}
    fL_max = {d: {i: {s: norm_powerL[di,ii,:,si].max() for si, s in enumerate(S)}
                 for ii, i in enumerate(I)} for di, d in enumerate(D)}
    A = {t: {i: {s: op_time[ti,ii,si] for si, s in enumerate(S)}
                 for ii, i in enumerate(I)} for ti, t in enumerate(T)}
    AL = {d: {i: {s: op_timeL[di,ii,si] for si, s in enumerate(S)}
                 for ii, i in enumerate(I)} for di, d in enumerate(D)}
    lambda_nom = {i: {s: RUL_nom[ii,si] for si, s in enumerate(S)}
                 for ii, i in enumerate(I)}
    F = {t: {i: {j: {s: RUL_LG[ti,ii,ji,si] for si, s in enumerate(S)} 
                 for ji, j in enumerate(J)} for ii, i in enumerate(I)} 
                 for ti, t in enumerate(T)}
    FL = {d: {i: {j: {s: RUL_LG_LTH[di,ii,ji,si] for si, s in enumerate(S)} 
                 for ji, j in enumerate(J)} for ii, i in enumerate(I)} 
                 for di, d in enumerate(D)}
    zetaL0 = {d:{i:{s: zeta_0[di,ii,si] for si, s in enumerate(S)} 
             for ii, i in enumerate(I)} for di, d in enumerate(D)}
    zeta0 = {i:{s:lambda_nom[i][s]>=1 for s in S} for i in I}
    t0 = {i: cycle_time[ii] for ii, i in enumerate(I)}
    C_max = {i: Fi/max([t0[i]-10,1]) for ii, i in enumerate(I)}
    CL_max = {d: {i: Fi/max([t0[i]+dd-10,1]) for ii, i in enumerate(I)} for dd, d in enumerate(D)}
    cr_coef = {i: Fi/CL_max[D[-1]][i] for ii, i in enumerate(I)} #maintenance cost criticality coefficient
    
    A_LTH = {d: {i: {s: (1 + (1-access_LTH[di,ii,si])) for si, s in enumerate(S)}
                 for ii, i in enumerate(I)} for di, d in enumerate(D)}

    
    
    ## Initiate model instance and declare variables ##
    owf = gp.Model('POSYDON')
    owf.reset(0)
    owf.update()

    # Continuous variables
    p = owf.addVars(T, I, S, lb=0, name="p") # hourly power generation for the STH
    pL = owf.addVars(D, I, S, lb=0, name="pL") # Daily power generation for the LTH
    l_STH = owf.addVar(lb=-np.inf, name = "l_STH") #Profit obtained in the STH
    l_LTH = owf.addVars(D, lb=-np.inf, name = "l_LTH") #Profit obtained in d d of the LTH
    lamda = owf.addVars(I, S, name = "lamda") #RUL
    ac = owf.addVars(I, S, lb=0, name = "ac") #Linearization variable for STH DMC 
    acL = owf.addVars(D, D, I, S, lb=0, name = "acL") #Linearization variable for LTH DMC
    a0cL = owf.addVars(D, I, S, lb=0, name = "a0cL") #Linearization variable for LTH DMC
    C = owf.addVars(I, lb=0, name = "C") #STH Dynamic Maintenance Cost (DMC)
    CL = owf.addVars(D, I, lb=0, name = "CL") #LTH Dynamic Maintenance Cost (DMC)
    amc = owf.addVars(T, I, lb=0, name = "amc") #Linearization variable for DMC in STH profit
    amcL = owf.addVars(D, I, S, lb=0, name = "amcL") #Linearization variable for DMC in LTH profit
    a_lambda = owf.addVars(I, S, lb=0, ub=0 if STH_sol is None else np.inf, name = "a_lambda") #penalty term
    
    
    
    # Integer variable
    q = owf.addVars(S,lb=0, vtype = gp.GRB.INTEGER, name = "q") # Overtime hours STH
    qL = owf.addVars(D, S, lb=0, vtype = gp.GRB.INTEGER, name = "qL") # Overtime hours LTH
    qa = owf.addVars(S,lb=0, ub=0 if STH_sol is None else np.inf, vtype = gp.GRB.INTEGER, name = "qa") #penalty term
    xa = owf.addVars(S,lb=0, ub=0 if STH_sol is None else np.inf, vtype = gp.GRB.INTEGER, name = "xa") #penalty term
    b = owf.addVars(I, S, lb = 0, vtype = gp.GRB.INTEGER, name = "b") # Remaining maintenance hours
    
    
    
    # Binary Variables 
    m = owf.addVars(T, I, vtype = gp.GRB.BINARY, name = "m") #STH maintenance schedule
    mL = owf.addVars(D, I, S, vtype = gp.GRB.BINARY, name = "mL") #LTH  maintenance schedule
    yaw = owf.addVars(T, I, J, vtype = gp.GRB.BINARY, name = "yaw") #STH YM decsions
    yawL = owf.addVars(D, I, J, S, vtype = gp.GRB.BINARY, name = "yawL") #LTH YM decisions
    zeta = owf.addVars(I, S, vtype = gp.GRB.BINARY, name = "zeta")  #WT STH operational status
    zetaL = owf.addVars(D, I, S, vtype = gp.GRB.BINARY, name = "zetaL") #WT LTH operational status
    y = owf.addVars(T, I, S, vtype = gp.GRB.BINARY, name = "y") #WT STH power availability
    yL = owf.addVars(D, I, S, vtype = gp.GRB.BINARY, name = "yL") #WT LTH power availability
    v = owf.addVar(vtype = gp.GRB.BINARY, name = "v") #STH vessel rental
    vL = owf.addVars(D, S, vtype = gp.GRB.BINARY, name = "vL") #LTH vessel rental
    x = owf.addVars(T, I, S, vtype = gp.GRB.BINARY, name = "x") #WT under maintenance
    w = owf.addVars(I, S, vtype = gp.GRB.BINARY, name = "w") #Prolonged maintenance
    z = owf.addVars(T, I, S, vtype = gp.GRB.BINARY, name = "z") #Downtime from ongoing maintenance
    theta = owf.addVars(I, vtype = gp.GRB.BINARY, name = "theta") #WT requires maintenance 
    
    
    
    ## Simulation mode ##
    if STH_sol is not(None):
        STH_sched = STH_sol[0]
        STH_yaw = STH_sol[1]
        print("*** Initiating day-ahead simulation from input schedule ***")
        for ii, i in enumerate(I):
            for tt, t in enumerate(T):
                m[t,i].lb = m[t,i].ub = STH_sched[tt,ii]
                for jj, j in enumerate(J):
                    yaw[t,i,j].lb = yaw[t,i,j].ub = STH_yaw[tt,ii,jj]

    else:
        print("##########################################################")
        print("# POSYDON: Yaw misalignment and maintenance optimization #")
        print("##########################################################")

    if TBS: 
        for ii, i in enumerate(I):
            theta[i].lb = theta[i].ub = theta_i[ii]
        #If a maintenance is needed, a vessel rental is enforced in the STH to 
        #incentivize (but not enforce) day-ahead maintenance
        if np.any(theta_i==1): v.lb = v.ub = 1
    
    #NOYAW or TBS or CBS Bencmark mode
    #We only allow 0 YM or shut down
    #Note: it is assumed that the middle j (i.e. J//2) is the 0 YM case
    if NOYAW | TBS | CMS:
        for i in I:
            for jj, j in enumerate(J):
                for t in T:
                    if jj!=len(J)//2: 
                        yaw[t,i,j].ub=0 
                for d in D:
                    for s in S:
                        if jj!=len(J)//2:
                            yawL[d,i,j,s].ub=0



    U = {i: {s: Omega+R*PiL[D[0]][s]*fL_max[D[0]][i][s]*tR for s in S}
         for i in I} #Upfront costs from not finishing an initiated task
    Y = {i: {s: R*PiL[D[0]][s]*fL_max[D[0]][i][s] for s in S} 
         for i in I} #hourly costs from not finishing an initiated task

    
    # Constraints
    obj_fun = l_STH + gp.quicksum(l_LTH[d] for d in D) - 1/N_S*(gp.quicksum(
        U[i][s]*w[i,s]+Y[i][s]*b[i,s]*A_LTH['d1'][i][s] for i in I for s in S) + C_lambda*gp.quicksum(lambda_nom[i][s]-lamda[i,s] +
                             lambda_nom[i][s]*gp.quicksum(m[t,i] for t in T) + gp.quicksum((lambda_nom[i][s]-D.index(d)-2)*mL[d,i,s]
                             for d in D) for i in I for s in S) + 30000*gp.quicksum(a_lambda[i,s]+qa[s]+xa[s] for i in I for s in S))  #objective function
    
    STH_dcf = owf.addConstrs((gp.quicksum(ac[i,s] for s in S) + C[i]*N_S*t0[i] == Kappa*gp.quicksum(zeta[i,s] for s in S) +
                              Fi*(N_S - gp.quicksum(zeta[i,s] for s in S)) for i in I), name="STH dcf")
    
    STH_dcf_L1 = owf.addConstrs((ac[i,s] <= C_max[i]*zeta[i,s] for i in I for s in S), name="STH dcf lin 1")
    
    STH_dcf_L2 = owf.addConstrs((ac[i,s] <= C[i] for i in I for s in S), name="STH dcf lin 2")
    
    STH_dcf_L3 = owf.addConstrs((ac[i,s] >= C[i] - C_max[i]*(1-zeta[i,s]) for i in I for s in S), name="STH dcf lin 3")
    
    LTH_dcf = owf.addConstrs((gp.quicksum(a0cL[d,i,s] for s in S) + gp.quicksum(acL[d1,d,i,s] for d1 in D[:D.index(d)+1] for s in S) + 
                              CL[d,i]*N_S*t0[i] == Kappa*gp.quicksum(zetaL[d,i,s] for s in S) +
                              Fi*(N_S - gp.quicksum(zetaL[d,i,s] for s in S)) for d in D for i in I), name="LTH dcf")
    
    LTH_dcf_L1 = owf.addConstrs((a0cL[d,i,s] <= CL_max[d][i]*zeta[i,s] for d in D for i in I for s in S), name="LTH dcf lin 1")
    
    LTH_dcf_L2 = owf.addConstrs((a0cL[d,i,s] <= CL[d,i] for d in D for i in I for s in S), name="LTH dcf lin 2")
    
    LTH_dcf_L3 = owf.addConstrs((a0cL[d,i,s] >= CL[d,i] - CL_max[d][i]*(1-zeta[i,s]) for d in D for i in I for s in S), name="LTH dcf lin 3")
    
    LTH_dcf_L4 = owf.addConstrs((acL[d1,d,i,s] <= CL_max[d][i]*zetaL[d1,i,s] for d in D for d1 in D[:D.index(d)+1] for i in I for s in S), name="LTH dcf lin 1")
    
    LTH_dcf_L5 = owf.addConstrs((acL[d1,d,i,s] <= CL[d,i] for d in D for d1 in D[:D.index(d)+1] for i in I for s in S), name="LTH dcf lin 2")
    
    LTH_dcf_L6 = owf.addConstrs((acL[d1,d,i,s] >= CL[d,i] - CL_max[d][i]*(1-zetaL[d1,i,s]) for d in D for d1 in D[:D.index(d)+1] for i in I for s in S), name="LTH dcf lin 3")
    
    STH_prof = owf.addConstr((l_STH == 1/N_S*gp.quicksum(gp.quicksum(Pi[t][s]*p[t,i,s]-Psi*x[t,i,s] for t in T for i in I)-Q*q[s] for s in S)
                             -gp.quicksum(cr_coef[i]*(1-rho[i])*amc[t,i] for t in T for i in I)-Omega*v), name = "STH profit")
    
    STH_prof_L1 = owf.addConstrs((amc[t,i] <= C_max[i]*m[t,i] for t in T for i in I), name="STH prof lin 1")
    
    STH_prof_L2 = owf.addConstrs((amc[t,i] <= C[i] for t in T for i in I), name="STH prof lin 2")
    
    STH_prof_L3 = owf.addConstrs((amc[t,i] >= C[i] - C_max[i]*(1-m[t,i]) for t in T for i in I), name="STH prof lin 3")
    
    LTH_prof = owf.addConstrs((l_LTH[d] == 1/N_S*gp.quicksum(gp.quicksum(PiL[d][s]*pL[d,i,s]-cr_coef[i]*(1-rho[i])*amcL[d,i,s]
                               -Psi*AL[d][i][s]*mL[d,i,s] for i in I)-Omega*vL[d,s]-Q*qL[d,s] for s in S) for d in D), 
                               name = "LTH profit")
    
    LTH_prof_L1 = owf.addConstrs((amcL[d,i,s] <= CL_max[d][i]*mL[d,i,s] for d in D for i in I for s in S), name="LTH prof lin 1")
    
    LTH_prof_L2 = owf.addConstrs((amcL[d,i,s] <= CL[d,i] for d in D for i in I for s in S), name="LTH prof lin 2")
    
    LTH_prof_L3 = owf.addConstrs((amcL[d,i,s] >= CL[d,i] - CL_max[d][i]*(1-mL[d,i,s]) for d in D for i in I for s in S), name="LTH prof lin 3")
    
    STH_stat1 = owf.addConstrs((zeta[i,s] <= lamda[i,s] for i in I for s in S), name = "STH WT status")
    
    STH_stat2 = owf.addConstrs((lamda[i,s] - 1 <= 500*zeta[i,s] for i in I for s in S), name = "STH WT status")
    
    LTH_stat1 = owf.addConstrs((lamda[i,s] - (D.index(d)+2) <= zetaL[d,i,s]*500 for d in D for i in I for s in S), name = "LTH WT status 1")
    
    LTH_stat2 = owf.addConstrs(((D.index(d)+2)*zetaL[d,i,s] <= lamda[i,s] for d in D for i in I for s in S), name = "LTH WT status 2")
    
    yaw_deg = owf.addConstrs((lamda[i,s] - a_lambda[i,s] == lambda_nom[i][s] + zeta0[i][s]*gp.quicksum(1-gp.quicksum(
                            yaw[t,i,j]*F[t][i][j][s] for j in J) for t in T)/24 + gp.quicksum(
                            zetaL0[d][i][s]*(1-gp.quicksum(yawL[d,i,j,s]*FL[d][i][j][s] for j in J)) for d in D)  
                                            for i in I for s in S), name = "Degradation rate")
    
    con41 = owf.addConstrs((gp.quicksum(m[t,i] for t in T)+gp.quicksum(mL[d,i,s] for d in D) == theta[i]
                           for i in I for s in S), name = "Force maintenance")
    
    con42 = owf.addConstrs((N_theta - lamda[i,s] <= theta[i]*1000 for i in I for s in S), name = "Force maintenance 2")
    
    con5 = owf.addConstrs((m[t,i] <= (T.index(t))/tR for t in T for i in I), 
                          name = "Maintenance after sunrise")
    
    con6 = owf.addConstrs((m[t,i] <= tD/(1.01+T.index(t)) for t in T for i in I), 
                          name = "Maintenance before sunset")
    
    con7 = owf.addConstrs((gp.quicksum(z[T[T.index(t)+t_hat],i,s] for t_hat in range(min([len(T[T.index(t):]),A[t][i][s]])))
                          >= min([len(T[T.index(t):]),A[t][i][s]])*m[t,i] for t in T for i in I for s in S
                          if T.index(t)+1.01 <= tD), name = "Downtime from ongoing maintenance")
    
    con8 = owf.addConstrs((b[i,s] >= gp.quicksum(m[t,i]*max([0,A[t][i][s]-len(T[T.index(t):])]) for t in T) for i in I for s in S), 
                          name = "Remaining hours of unfinished maintenance")
    
    con9 = owf.addConstrs((w[i,s] >= b[i,s]/100 for i in I for s in S), name = "Unfinished maintenance")
    
    con10 = owf.addConstrs((x[t,i,s] >= z[t,i,s]-(T.index(t))/tD for t in T for i in I for s in S), 
                           name = "Crew occupacy")
    
    con11 = owf.addConstrs((gp.quicksum(x[t,i,s] for i in I) <= B + xa[s]  for t in T for s in S), 
                           name = 'Max tasks per t')    
    
    con16 = owf.addConstrs((y[t,i,s] <= zeta[i,s]*(1-rho[i])+ gp.quicksum((len(T[T.index(tt):]))*m[tt,i] for tt in T)/
                             (len(T[T.index(t):])+0.1) for t in T for i in I for s in S), name = 'Availability STH')
    
    con17 = owf.addConstrs((yL[d,i,s] <= zetaL[d,i,s]*(1-rho[i])+(N_D-gp.quicksum((D.index(dd)+1)*mL[dd,i,s] for dd in D))/
                            (N_D-D.index(d)-0.9) for d in D for i in I for s in S), name = "Availability LTH")
    
    con18 = owf.addConstrs((y[t,i,s] <= 1 - z[t,i,s] for t in T for i in I for s in S),
                           name = "Unavailability from maintenance")
    
    con19 = owf.addConstr((v>=1/N_t*gp.quicksum(m[t,i] for t in T for i in I)), 
                           name = "STH vessel rental")
    
    con20 = owf.addConstrs((vL[d,s] >= 1/N_t*gp.quicksum(mL[d,i,s] for i in I) 
                            for d in D for s in S), name = "LTH vessel rental")
    
    con21 = owf.addConstrs((gp.quicksum(x[t,i,s] for t in T for i in I)<=B*W+q[s]+qa[s] for s in S), 
                          name = "Overtime")
    
    con22 = owf.addConstrs((gp.quicksum(mL[d,i,s]*AL[d][i][s]+b[i,s] for i in I) <= B*W+qL[d,s]
                            for d in D for s in S if D.index(d)==0), name = 'Overtime 1st day of LTH')
    
    con23 = owf.addConstrs((gp.quicksum(mL[d,i,s]*AL[d][i][s] for i in I) <= B*W+qL[d,s] for d in D for s in S if D.index(d)>0), 
                           name = 'Overtime other days of LTH')
    
    con24 = owf.addConstrs((q[s] <= H for s in S), name = "STH max overtime")
    
    con25 = owf.addConstrs((qL[d,s] <= H for d in D for s in S), name = "LTH max overtime")
    
    yaw_pwr = owf.addConstrs((p[t,i,s] <= R*gp.quicksum(yaw[t,i,j]*f[t][i][j][s] for j in J) for t in T for i in I for s in S), name = "STH power")
    
    con26 = owf.addConstrs((p[t,i,s] <= R*y[t,i,s] for t in T for i in I for s in S), name = "STH power")
    
    yaws = owf.addConstrs((gp.quicksum(yaw[t,i,j] for j in J) <= 1 for t in T for i in I), name = "STH yaw")
    
    yawsL = owf.addConstrs((gp.quicksum(yawL[d,i,j,s] for j in J) <= 1 for d in D for i in I for s in S), name = "LTH yaw")
    
    con27 = owf.addConstrs((pL[d,i,s] <= 24*R*gp.quicksum(yawL[d,i,j,s]*fL[d][i][j][s] for j in J) 
                            for d in D for i in I for s in S), name = "LTH power 1")
    
    con270 = owf.addConstrs((pL[d,i,s] <= 24*R*yL[d,i,s] for d in D for i in I for s in S), name = "LTH power 2")
    
    con271 = owf.addConstrs((pL[d,i,s] <= R*fL_max[d][i][s]*(24 - AL[d][i][s]*mL[d,i,s]) 
                            for d in D for i in I for s in S), name = "LTH power 2")
    
    if CMS:
        assert len(S)==1, 'CBS is a deterministic benchmark.'
        cms_con = owf.addConstrs((gp.quicksum(m[t,i] for t in T) <= 1-zeta0[i][s] for i in I for s in S), 
                           name = 'CMS benchmark constraint')
    
        

    #########################################################################################################


    # Set objective
    owf.setObjective(obj_fun, gp.GRB.MAXIMIZE)

    owf.setParam("MIPGap", rel_gap)
    owf.setParam("TimeLimit", 3600/2)

    owf.update()
    

    # Solve model
    owf.optimize()
    
    
    if owf.solCount == 0:
        #try to solve one more time with relaxed feasibility tolerance
        owf.reset(0)
        print('Infeasibility encountered; relaxing Feasibility Tolerance')
        owf.setParam("FeasibilityTol", 1e-2)
        owf.setParam("MIPGap", rel_gap)
        owf.setParam("TimeLimit", 3600/2)
        owf.optimize()
        if owf.solCount == 0:
            owf.computeIIS()
            return None
        
    STH_sched = np.round(np.array(owf.getAttr('X',m).values()).reshape(-1,N_t))
    
    STH_yaw = np.round(np.array(owf.getAttr('X',yaw).values()).reshape(-1,N_t,N_J))

    LTH_sched = np.array(owf.getAttr('X',mL).values()).reshape(-1,N_t,N_S)
            
    DMC = np.concatenate((np.array(owf.getAttr('X',C).values()).reshape(1,-1),
                          np.array(owf.getAttr('X',CL).values()).reshape(N_D,N_t)), axis=0)
    
    #need to attend to expected costs (max power output)
    expected_STH_cost = (1/N_S*np.sum([np.array([f[t][i][j][s] for j in J]).max()*R*Pi[t][s] for t in T for i in I for s in S])-l_STH.X)
    
    var_name = "yawL"
    model_dict =  {var.varName: owf.getVarByName(var.varName).X for var in owf.getVars()} 
    var_value_list = []
    for var, value in model_dict.items():
        if var_name+'[' in var:
            var_value_list.append(value)
    
    LTH_yaw = np.array(var_value_list).reshape(N_D,N_t,N_J,N_S)
    
    var_name = "lamda"
    model_dict =  {var.varName: owf.getVarByName(var.varName).X for var in owf.getVars()} 
    var_value_list = []
    for var, value in model_dict.items():
        if var_name+'[' in var:
            var_value_list.append(value)
    lambda_is = np.array(var_value_list).reshape(N_t,N_S)
    
    output = {'STH_sched': STH_sched, 
              'STH_yaw': STH_yaw,
              'LTH_sched': LTH_sched, 
              'LTH_yaw': LTH_yaw,
              'DMC': DMC,
              'crit_coef': np.array(list(cr_coef.values())),
              'lambda_is': lambda_is,
              'expected_STH_cost': expected_STH_cost,
              'remaining_maint_hrs': np.array(owf.getAttr('X',b).values()).reshape(N_t,N_S),
              'other_params': {'mip_gap':owf.mip_gap,
                               },
              'model': model_dict if return_all_vars else None
             }
    return output



#%%

def quadKernelAugmentedSpace(X):
    """
    Maps input space of shape (obs_dim, 2) to augmented space of shape (obs_dim, 6) of a quadratic Kernel.
    
    The first column of X is the daily avg wind speed and the second column is the daily avg wave height.
    """
    out = np.append(np.sqrt(2)*X,X**2,1)
    out = np.append(out, (np.sqrt(2)*X[:,0]*X[:,1]).reshape(-1,1),1)
    out = np.append(out, np.ones((X.shape[0],1)) ,1)
    return out

#%%

def missionTimeLTH(wind_speed_forecast_LTH, wave_height_forecast_LTH, tau, wave_lim=1.5, coefs=None):
    """
    Returns the expected hours of mission time for each day of the LTH, given the
    average daily wind speed and wave height
    
    Inputs:
    - wind_speed_forecast_LTH: np float array of shape (N_D, N_t, N_S)
    - wave_height_forecast_LTH: np float array of shape (N_D, N_S)
    - tau: np int array of shape (N_t,)
    - wave_lim: wave height safety threshold (1.5m, 1.8m or 2m) to use precomputed values of regression parameters.
    - coefs: np float arry of shape (1,6) -> coefficients of the regression using quadratic kernel space. 
        If the default value of None is used, one of the 3 precomputed values will be used for wave_lim.
    
    Returns:
    - mission_time: np float array of shape (N_D, N_t, N_S) -> LTH mission time scenarios
    - access_LTH: np float array of shape (N_D, N_t, N_S) -> daily accessibility fraction scenarios
    """
    (N_D, N_t, N_S) = wind_speed_forecast_LTH.shape
    #weights of augmented space from RidgeRegression(0.0001) with quadratic Kernel 
    #using historical NWP data and true accessibility data, to calculate the 
    #percentage of hours in a day of the LTH where the turbine will be inaccessible
    #Posted here to slightly reduce the complexity
    if coefs==None:
        if wave_lim==1.5:
            b = np.array(  [[ 0.02008826],
                            [0.76945388],
                            [-0.00409112],
                            [ 0.06934806],
                            [ 0.02252065],
                            [ 1.68621608]])  
        elif wave_lim==1.8:
            b = np.array(  [[ 0.03513277],
                            [-0.58401405],
                            [-0.00536439],
                            [ 0.02710852],
                            [ 0.02153183],
                            [ 1.47014811]])
        elif wave_lim==2:
            b = np.array(  [[ 4.18107897e-02],
                            [-4.62873380e-01],
                            [-6.23088680e-03],
                            [-7.37855258e-04],
                            [ 2.19760977e-02],
                            [ 1.32768890e+00]])
    else: b=coefs
    
    mission_time = np.zeros((N_D, N_t, N_S))
    access_LTH = np.zeros((N_D, N_t, N_S))
    for i in range(N_t):
        for s in range(N_S):
            X_in = np.append(wind_speed_forecast_LTH[:,i,s].reshape(-1,1),
                             wave_height_forecast_LTH[:,s].reshape(-1,1), 1)
            X = quadKernelAugmentedSpace(X_in)
            
            access_LTH[:,i,s] = np.matmul(X.copy(),b).reshape(-1) #(N_D,)
            access_LTH[access_LTH<0] = 0
            access_LTH[access_LTH>1] = 1
            #mission_time = repair_time + {frac of repair time that will be inaccessible}
            #mission_time = repair_time + repair_time*(1-access)
            mission_time[:,i,s] = tau[i]*(1 + (1-access_LTH[:,i,s]))
            
    
    return mission_time, access_LTH
            
#%%

def tsScenarioGen(
        forecast_error_hist,
        pred_len,
        N_S,
        custom_hist_len = 24*10,
        random_seed=1
        ):

    """
    Scenario generation for deterministic time series forecast errors
    sklearn.gaussian_process.GaussianProcessRegressor fitting and sampling.
    
    Inputs:
    - forecast_error_hist: np float array of shape (hist_len,) -> Full error history
    - pred_len: int -> prediction length
    - N_S: int -> Number of scenarios to generate
    - *Optional*
        - custom_hist_len: int -> Custom history length. If hist_len<custom_hist_len,
            then hist_len is used (default=24*10)
        - random_seed=1
        
    Returns:
    -forecast_error_scenarios: np float array of shape (pred_len, N_S)
    """
    
    hist_len = forecast_error_hist.shape[0]
    
    forecast_error_scenarios = np.zeros((pred_len, N_S))
    
    if custom_hist_len>hist_len: custom_hist_len=hist_len
    
    kernel = RBF()
    gp_model = GaussianProcessRegressor(
            kernel=kernel, 
            alpha = np.random.uniform(size=(custom_hist_len))**2,
            random_state=random_seed,  
            n_restarts_optimizer=10)
    
    X = np.arange(custom_hist_len).reshape(-1,1)
    Y = forecast_error_hist[hist_len-custom_hist_len:]
    gp_model.fit(X, Y)
    
    X_pred = (X[-1]+np.arange(1,pred_len+1)).reshape(-1,1)
    
    if np.all(Y==0):
        forecast_error_scenarios = np.zeros((pred_len, N_S))
    #Old benchmark:
    #elif DGP:
    #    mean_y, cov_y = gp_model.predict(X_pred, return_cov=True)
    #    forecast_error_scenarios = mean_y.reshape(-1,1)
    else:
        scenario_set = gp_model.sample_y(X_pred,N_S,random_state=random_seed)
        std_gen = scenario_set.std(-1)
        std_hist = forecast_error_hist.std()
        scenario_scales = std_gen/std_hist
        forecast_error_scenarios = scenario_set/scenario_scales.reshape(-1,1)
        
    return forecast_error_scenarios


#%%
    
def equivalentTimeTransform(wind_speed_scenarios, loading_map):
    """
    Generates yaw- and wind-dependent time loss/gain scenarios based on the loading map from Kragh (2014)
    
    Inputs: 
    - wind_speed_scenarios: numpy array of shape (time_dim, N_I, N_S); the wind speed scenarios for each WT
    - loading_map: pandas DataFrame of shape (N_J, wind_bins); the loading map, whose indexes and columns are lists
        of the upper and lower bounds [lb, ub] of the bins for the yaw misalignment levels and wind speed, respectively. 
        Yaw misalignment levels should be in decreasing order (e.g. 40deg to -40deg) and wind speed levels should be in 
        increaing order (e.g. 6m/s to 20m/s).
        
    Returns:
    - equivalent_time: numpy array of shape (time_dim, N_I, N_J, N_S)
    """
    
    time_dim, N_I, N_S = wind_speed_scenarios.shape
    N_J, wind_bins = loading_map.shape
    
    if type(loading_map.index[0]) == str:
        ym_range = [] 
        for i in loading_map.index:
            ym_range.append(eval(i))
    else: 
        ym_range = loading_map.index
    
    if type(loading_map.columns[0]) == str:
        ws_range = []
        for i in loading_map.columns:
            ws_range.append(eval(i))
    else: 
        ws_range = loading_map.columns

    N2 = np.full((time_dim, N_I, N_J, N_S), np.inf)
    temp = np.repeat(np.repeat(np.repeat(np.array(loading_map)[:,:,np.newaxis], N_S, -1)[np.newaxis,:,:,:],
                     time_dim, 0)[:,np.newaxis,:,:,:], N_I, 1) #(time_dim, N_I, N_J, wind_bins, N_S)
    ws_yaw = np.repeat(wind_speed_scenarios[:,:,np.newaxis,:], N_J, 2)
    for i, ws in enumerate(ws_range):
        bool_map = (ws_yaw>ws-1) & (ws_yaw<ws)
        N2[bool_map] = temp[:,:,:,i,:][bool_map]
        
    #equivalent_time = 1-1/N2
    
    return 1/N2



#%%


def YawDependentPowerCurve(wind_speed, transformation_map):
    """
    Performs the binning method to calculate the normalized power from wind speed time series data.
    
    Inputs:
    - wind_speed: np.array of floats of shape (time_dim, N_t, N_S) -> Wind speed scenarios
    - tranformation_map: pd.DataFrame of shape (N_J, wind_bins) 
        
    Returns:
    - norm_power: np.array of the same shape as wind_speed containing the normalized power values
    """
    
    (time_dim, N_t, N_S) = wind_speed.shape
    N_J = transformation_map.shape[0]
    
    norm_power = np.zeros((time_dim, N_t, N_J, N_S))
    norm_power2 = np.transpose(norm_power, (0,1,3,2)) #(time_dim, N_t, N_S, N_J)

    for j in transformation_map.columns:
        index_true_STH = (wind_speed>=float(j)-1) & (
            wind_speed<float(j))
        norm_power2[index_true_STH,:] = np.array(transformation_map[j])
    
    return norm_power 


#%%

def missionTimeSTH(wind_speed,
                        wave_height,
                        vessel_max_wave_height,
                        vessel_max_wind_speed,
                        tau,
                        tR=5,
                        tD=21
                        ):
    
    _,N_t,N_S = wind_speed.shape
    
    # Hourly accessibility
    access = np.zeros_like(wind_speed)
    access[(wind_speed<vessel_max_wind_speed) & (
        np.repeat(wave_height.reshape(-1,1,N_S),N_t,axis=1)<vessel_max_wave_height)] = 1
    access2 = access.reshape(-1,24,N_t,N_S)
    access2[:,:tR,:,:] = 0 #sunrise constraint
    access2[:,tD:,:,:] = 0 #sunset constraint
    
    # Calculate operation times
    op_time = np.zeros((wind_speed.shape), dtype=int)
    op_time2 = op_time.reshape(-1,24,N_t,N_S)
    
    
    for i in range(N_t):
        for t in range(24):
            temp = op_time2[:,t,i,:]
            op_time2[:,t,i,:] = (np.cumsum(access2[:,t:,i,:],1)>=tau[i]).argmax(1)
            temp[temp==0] = (23-t+tau[i]-np.sum(access2[:,t:,i,:],1))[temp==0]
    
    op_time2+=1
    
    return op_time2[0,:,:,:], access2[0,:,:,:]


#%%


def dataLoader(ws_STH,  
                ws_LTH,
                wh_STH,
                wh_LTH,
                ep_STH,
                ep_LTH, 
                ws_err_hist,
                wh_err_hist,
                ep_err_hist,
                N_S,
                max_wind, 
                max_wave,
                tau,
                power_curve_map,
                loading_map,
                tR=5, 
                tD=21,
                hist_len=5,
                random_seed=1,
                sim_day_ahead=False,
                NAIVE=False):
    """
    A function used to prepare the inputs of POSYDON using forecasts and historical errors.
    
    Inputs:
    - ws_STH: np.array of shape (24,N_t) -> Hourly wind speed data for each wind turbine in the STH
    - ws_LTH: np.array of shape (N_D,N_t) -> Daily average wind speed data for each wind turbine in the LTH
    - wh_STH: np.array of shape (24,) -> Hourly wave height data for STH
    - wh_LTH: np.array of shape (N_D,) -> Daily average wave height data for LTH
    - ep_STH: np.array of shape (24,) -> Hourly electricity prices data for STH
    - ep_LTH: np.array of shape (N_D,) -> Daily average electricity price data for LTH
    - ws_err_hist: np.array of shape (hist_len,N_t) -> Historical hourly wind speed forecast error data for each 
        turbine
    - wh_err_hist: np.array of shape (hist_len,) -> Historical hourly wave height forecast error data
    - ep_err_hist: np.array of shape (hist_len,) -> Historical hourly electricity price forecast error data
    - N_S: integer -> Number of scenarios to generate
    - max_wind: float -> Safety threshold for maximum wind speed
    - max_wave: floar -> Safety threshold for maximum wave height
    - rle_mean: np.array of shape (N_t,) -> Mean of residual life estimates for each wind turbine
    - rle_std: float or np.array of floats of shape (N_t,) -> RLE forecast STD used in random scenario generator
    - tau: integer np.array of shape (N_t,) -> Workhours required to complete each maintenance task
    - rle_dist: (optional) str -> the distibution to be used for the scenario generation. Valid inputs are 'normal' and 'weibull'.
        If 'weibull' is selected, the rle_std is used as the shape parameter, and rle_mean as the scale.
    - tR: (optional) integer -> Time of first sunlight in 24-hour basis (default=5) 
    - tD: (optional) integer -> Time of last sunlight in 24-hour basis (default=21) 
    - hist_len (optional) integer -> Custom history length for scenario generation in days (default=5)
    - random_seed: (optional) integer -> Seed used for reproducibility of scenario generation (default=1)
    - sim_day_ahead: (optional) bool -> Simulate the day ahead using a predefined schedule and true data/no scenarios 
        (default=False)
    - NAIVE: (optional) bool -> use a naive scenario generation method (Default=False)
        
    Returns:
    - norm_power: np.array of shape (24, N_t, N_S) -> hourly normalized power scenarios of the wind turbines in the STH
    - norm_powerL: np.array of shape (N_D, N_t, S) -> daily normalized power scenarios of the wind turbines in the LTH
    - el_price: np.array of shape (24, N_S) -> hourly electricity price scenarios in the STH [$/MWh]
    - el_priceL: np.array of shape (N_D, N_S) -> daily electricity price scenarios in the LTH [$/MWh]
    - op_time: np.array of shape (24, N_t, N_S) -> hourly operation time scenarios for turbine maintenance [hours]
    - op_timeL: np.array of shape (N_D, N_t, N_S) -> daily operation time scenarios for turbine maintenance [hours]
    - rle: np.array of shape (N_t, N_S) -> Scenarios for RLE
    - RUL_LG: np.array of shape (24, N_I, N_J, N_S)  -> RUL loss/gain for the STH decisions
    - RUL_LG_LTH: np.array of shape N_D, N_I, N_J, N_S) -> RUL loss/gain for the LTH decisions
    - access: np.array of shape (24, N_t, N_S) -> STH accessibility
    - access_LTH: np.array of shape (N_D, N_t, N_S) -> LTH accessibility
    - ws_STH_scenarios: np.array of shape (24, N_t, N_S) -> STH wind speed scenarios
    - wh_STH_scenarios: np.array of shape (24, N_S) -> STH wave height scenarios 
    - ws_LTH_scenarios: np.array of shape (N_D, N_t, N_S) -> LTH wind speed scenarios 
    - wh_LTH_scenarios: np.array of shape (N_D, N_S) -> LTH wave height scenarios
    """
    N_D, N_t = ws_LTH.shape
    
    t1=time.time()
    
    if (sim_day_ahead) | (N_S==1):
        ws_STH_scenarios = ws_STH[:,:,np.newaxis]
        wh_STH_scenarios = wh_STH[:,np.newaxis]
        ws_LTH_scenarios = ws_LTH[:,:,np.newaxis]
        wh_LTH_scenarios = wh_LTH[:,np.newaxis]
        
        el_price = ep_STH[:,np.newaxis]
        el_priceL = ep_LTH[:,np.newaxis]

    else:
        ws_STH_scenarios = np.repeat(ws_STH[:,:,np.newaxis], N_S, -1,) 
        ws_LTH_scenarios = np.repeat(ws_LTH[:,:,np.newaxis], N_S, -1,) 
        
        wh_STH_scenarios = np.repeat(wh_STH[:,np.newaxis], N_S, -1,) 
        wh_LTH_scenarios = np.repeat(wh_LTH[:,np.newaxis], N_S, -1,) 
        
        el_price = np.repeat(ep_STH[:,np.newaxis], N_S, -1,) 
        el_priceL = np.repeat( ep_LTH[:,np.newaxis], N_S, -1,) 
        
        if NAIVE:
            np.random.seed(random_seed)
            for i in range(N_t):
                ws_STH_scenarios[:,i,:] -= np.random.normal(loc=np.zeros((24,1)),
                                scale=np.full((24,1),ws_err_hist[:,i].std()),size=(24,N_S))
                ws_LTH_scenarios[:,i,:] -= np.random.normal(loc=np.zeros((N_D,1)),
                                scale=np.full((N_D,1),ws_err_hist[:,i].std()),size=(N_D,N_S))
            
                wh_STH_scenarios -= np.random.normal(loc=np.zeros((24,1)),
                                scale=np.full((24,1),wh_err_hist.std()),size=(24,N_S))
                wh_LTH_scenarios -= np.random.normal(loc=np.zeros((N_D,1)),
                                scale=np.full((N_D,1),wh_err_hist.std()),size=(N_D,N_S))
                
                el_price -= np.random.normal(loc=np.zeros((24,1)),
                                scale=np.full((24,1),ep_err_hist.std()),size=(24,N_S))
                el_priceL -= np.random.normal(loc=np.zeros((N_D,1)),
                                scale=np.full((N_D,1),ep_err_hist.std()),size=(N_D,N_S))
        else:
            for i in range(N_t):
                ws_STH_scenarios[:,i,:] -= tsScenarioGen(ws_err_hist[:,i], 24, N_S, 
                      24*hist_len, random_seed)
                ws_LTH_scenarios[:,i,:] -= tsScenarioGen(ws_err_hist[:,i].reshape(-1,24).mean(1),
                                          N_D+1, N_S, 50, random_seed)[1:,:] #discard the day-ahead
        
            wh_STH_scenarios -= tsScenarioGen(wh_err_hist, 24, N_S, 24*hist_len, 
                  random_seed)
            wh_LTH_scenarios -= tsScenarioGen(wh_err_hist.reshape(-1,24).mean(1),
                                          N_D+1, N_S, 50, random_seed)[1:,:] #discard the day-ahead
            
            el_price -= tsScenarioGen(ep_err_hist, 24, N_S, 24*hist_len, random_seed)
            el_priceL -= tsScenarioGen(ep_err_hist.reshape(-1,24).mean(1),
                                          N_D+1, N_S, 50, random_seed)[1:,:] #discard the day-ahead
        
        ws_STH_scenarios[ws_STH_scenarios<0.5]=0.5
        wh_STH_scenarios[wh_STH_scenarios<0.1]=0.1
        ws_LTH_scenarios[ws_LTH_scenarios<0.5]=0.5
        wh_LTH_scenarios[wh_LTH_scenarios<0.1]=0.1
        
    
    norm_power = YawDependentPowerCurve(ws_STH_scenarios, power_curve_map)
    norm_powerL = YawDependentPowerCurve(ws_LTH_scenarios, power_curve_map)
    
    RUL_LG = equivalentTimeTransform(ws_STH_scenarios, loading_map)
    RUL_LG_LTH = equivalentTimeTransform(ws_LTH_scenarios, loading_map)
    
    op_time, access = missionTimeSTH(
            ws_STH_scenarios,
            wh_STH_scenarios,
            max_wave,
            max_wind,
            tau,
            tR,
            tD)
    

    op_timeL, access_LTH = missionTimeLTH(ws_LTH_scenarios, 
                                                wh_LTH_scenarios, 
                                                tau)

        
    print('Data prepared in ', round(time.time()-t1, 4), ' sec')
    
    return norm_power, norm_powerL, el_price, el_priceL, op_time, op_timeL, RUL_LG, RUL_LG_LTH, access, access_LTH, ws_STH_scenarios, wh_STH_scenarios, ws_LTH_scenarios, wh_LTH_scenarios


#%%


def getVarFromModel(model_dict, var_name, STH_len=24, LTH_len=19, N_t=5):
    """
    Function to help extract the model variables stored as dictionary items into numpy arrays.
    
    Inputs:
    - model_dict: A dictionary whose keys are the single variable names of the model in string format 
        ('var_name[time_idx,wt_idx,scenario_idx]') that stores the corresponding variable value.
    - var_name: A string with the name of the block of variables. 
    - STH_len: (optional; Default=24) The length of the STH.
    - LTH_len: (optional; Default=19) The length of the LTH.
    - N_t: (optional; Default=5) The number of wind turbines considered.
    
    Returns:
    - var_value_array: A numpy array with the values of the querried variable.
    - var_name_array: A numpy array with the names of each single variable.
    """
    var_name_list = []
    var_value_list = []
    for var, value in model_dict.items():
        if var_name+'[' in var:
            var_name_list.append(var)
            var_value_list.append(value)
    
    var_name_array = np.array(var_name_list)
    var_value_array = np.array(var_value_list)
    
    if '[d' in var_name_array[0]:
        if 'wt' in var_name_array[0]:
            if ',s' in var_name_array[0]:
                var_name_array = var_name_array.reshape(LTH_len,N_t,-1)
                var_value_array = var_value_array.reshape(LTH_len,N_t,-1)
            else:
                var_name_array = var_name_array.reshape(LTH_len,N_t)
                var_value_array = var_value_array.reshape(LTH_len,N_t)
        else:
            if ',s]' in var_name_array[0]:
                var_name_array = var_name_array.reshape(LTH_len,-1)
                var_value_array = var_value_array.reshape(LTH_len,-1)
            else:
                var_name_array = var_name_array.reshape(LTH_len)
                var_value_array = var_value_array.reshape(LTH_len)
                
    elif '[t' in var_name_array[0]:
        if 'wt' in var_name_array[0]:
            if ',s' in var_name_array[0]:
                var_name_array = var_name_array.reshape(STH_len,N_t,-1)
                var_value_array = var_value_array.reshape(STH_len,N_t,-1)
            else:
                var_name_array = var_name_array.reshape(STH_len,N_t)
                var_value_array = var_value_array.reshape(STH_len,N_t)
        else:
            if ',s' in var_name_array[0]:
                var_name_array = var_name_array.reshape(STH_len,-1)
                var_value_array = var_value_array.reshape(STH_len,-1)
            else:
                var_name_array = var_name_array.reshape(STH_len)
                var_value_array = var_value_array.reshape(STH_len)
    
    elif '[wt' in var_name_array[0]:
        var_name_array = var_name_array.reshape(N_t,-1)
        var_value_array = var_value_array.reshape(N_t,-1)
    else:
        var_name_array = var_name_array.reshape(-1,1)
        var_value_array = var_value_array.reshape(-1,1)
    
    if var_name=='v':
        return np.array(model_dict['v']), np.array(['v'])
    
    else:
        return var_value_array, var_name_array


#%%


def solveRollingHorizon(wind_speed_true,
                    wave_height_true,
                    wind_speed_forecast,
                    wave_height_forecast,
                    N_S,
                    Din,
                    el_prices_true,
                    el_prices_forecast,
                    degr_experiment_num=1,
                    max_wind=15, 
                    max_wave=1.8,
                    deg_data_folder="Data/Degradation Signal info/Data for Optimization",
                    power_curve_folder="Data/Yaw power curve",
                    tR=5, 
                    tD=21,
                    N_D=19,
                    hist_len=5,
                    random_seed=1,
                    max_iter=None,
                    mip_gap=0.001,
                    tasks_per_wt=None,
                    CMS=False,
                    NOYAW=False,
                    TBS=False,
                    NAIVE=False):
    """
    Function that uses the rolling horizon iterative algorithm that gets the output of the stochastic model and uses it as
    input to a simulation model that uses true data to get the true schedule cost.
    
    Inputs:
    -wind_speed_true: np.array of shape (obs_dim, N_t) -> The true wind speed profile that will be used in the simulation 
         of the day-ahead
    -wave_height_true: np.array of shape (obs_dim, N_t) -> The true wave height profile that will be used in the simulation 
         of the day-ahead
    -wind_speed_forecast: np.array of shape (obs_dim, N_t) -> The forecasted wind speed profile that will be used in the 
        stochastic program
    -wave_height_forecast: np.array of shape (obs_dim, N_t) -> The forecasted wave height profile that will be used in the 
        stochastic program
    -N_S: int -> Number of scenarios to generate
    -Din: int -> Day of the dataset that will be the first day
    -el_prices_true: np.array of shape (time_dim, ) -> The true electricity price profile that will be used in the simulation 
         of the day-ahead
    -el_prices_forecast: np.array of shape (time_dim, ) -> The forecasted electricity price profile that will be used in the 
        stochastic program
    
    Benchmarks:
    -CMS: Corrective (reactive) maintenance strategy
    -NOYAW: Same model but perfect yaw alignment is assumed
    -TBS: Time-based strategy
    -NAIVE: Marginal densities for the scenario generation
    
    
    Optional:
    - degr_experiment_num: int (def=1) -> The experimental setup case for the degradation model
    - max_wind: float -> Maximum wind speed tolerance for vessel/crew operation (default=15)
    - max_wave: float -> Maximum wave height tolerance for vessel/crew operation (default=1.8)
    - deg_data_folder: str -> Degradation data filepath (Default="Data/Degradation Signal info/Data for Optimization")
    - power_curve_folder: str -> Power curve model filepath (Default="Data/Yaw power curve")
    - tR: int (default=5) -> Time of first daylight 
    - tD: int (default=21) -> Time of last daylight 
    - N_D: int (default=19) -> LTH length in days
    - hist_len: int (default=5) -> History length used in GPR for scenario generation in days
    - random_seed: int (default=1) -> Random seed
    - max_iter: int (default=None) -> Maximum number of iterations. If None is selected, the horizon rolls until all wind
        turbines have a completed maintenance task
    - mip_gap: float (default=0.001) -> Optimality maximum relative gap for the stochastic solution
    
    Returns:
    -output: A dictionary with items:
    'stochastic_sol': a list whose items are the dict outputs of POSYDON at each horizon roll
    'stochastic_input': a tuple whose elements are lists of the stochastic inputs to POSYDON at each horizon roll. Each list has
        inputs with the following order:
            0. norm_power
            1. norm_powerL
            2. el_price
            3. el_priceL
            4. op_time
            5. op_timeL
            6. access
            7. ws_STH
            8. wh_STH
            9. rle
            10. theta
            11. rho
    'simulation': a list whose items are the dict outputs of the simulation at each horizon roll using the STH schedule
        of 'stochastic_sol'
    'true_input': a tuple whose elements are lists of the true inputs to the simulation model at each horizon roll, following
        the same order as the 'stochastic_input'
    'time_per_roll': a list with the optimization time of each horizon roll
    'equivalent_times': np.array of shape (num_rolls, N_t) with the equivalent times (how far in the degradation process the turbines are)
    """
    
    obs_dim, N_t = wind_speed_true.shape
    total_days = obs_dim/24
    
    stochastic_input = []
    stochastic_output = []
    
    true_input = []
    true_output = []
    one_iteration_time = []
    
    rho = np.zeros(N_t)
    
    eq_times = [] #list of equivalent times
    
    task_num = np.zeros(N_t, dtype=int) #Number of tasks completed
    repair_times = np.array(pd.read_csv(deg_data_folder+'/Repair_times.csv', header=0, index_col=0, usecols=[
            j for j in range(N_t+1)])) #+1 accounts for index col
    tau = repair_times[task_num,np.arange(N_t)].copy()
    prev_tn = task_num.copy() #previous task number
    #select initial turbine ages (they are equivalent ages, will be used to select row):
    t_0 = np.array(pd.read_csv(deg_data_folder+"/WTs_initial_ages_Profiles/WTs_initial_age-"+str(degr_experiment_num)+".csv",header=0))[0,:N_t]
    t_0[t_0<1]=1 #t_0=0 would lead to infeasibilities
    #Get the degradation profile names (str) to select the appropriate dataset in each case:
    deg_profiles = np.array(pd.read_csv(deg_data_folder+"/WTs_initial_ages_Profiles/WTs_Signals-"+str(degr_experiment_num)+".csv",header=0,
                                        usecols=[j for j in range(N_t)])) #no index_col in this file
    deg_profile = deg_profiles[task_num, np.arange(N_t)]
    
    RUL_nominals_list = [] 
    RUL_true0 = np.zeros((N_t,1)) #The true RUL at the beginning of the degradation process
    for i in range(N_t):
        if N_S>1:
            RUL_nominals_list.append(np.array(
                    pd.read_csv(deg_data_folder+"/Failuretime_Scenarios_float/FailureT_scenarios_"+deg_profile[i]+".csv", header=None, usecols=[
                            j for j in range(N_S)]))) #no index cols or header
        else: 
            RUL_nominals_list.append(np.array(
                    pd.read_csv(deg_data_folder+"/Failuretime_Scenarios_float/FailureT_scenarios_"+deg_profile[i]+".csv", header=None)).mean(-1).reshape(-1,1))
        RUL_true0[i,0] = RUL_nominals_list[-1].shape[0] #shape=(N_I,1)
    
    #In the first iter, the true RUL is the same as the initial true RUL, minus the initial age
    RUL_true = RUL_true0.copy()-t_0.reshape(-1,1).copy() #shape=(N_I, 1)
    RUL_true[RUL_true<0]=0 #just in case
    
    equivalent_time = 0.+t_0 #Initially the equivalent time is the same as t_0 (-1 for 0-based indexing)
    
    eq_times.append(equivalent_time.copy()) #store them at each iteration to process results later
    
    power_curve_map = pd.read_csv(power_curve_folder+"/Yaw_dependent_power_curve.csv", header=0, index_col=0)
    
    loading_map = pd.read_csv(deg_data_folder+"/RUL_loss_gain_table.csv", header=0, index_col=0)

    run=0
    while run+Din+N_D+1<=total_days:
        
        if tasks_per_wt != None:
            if np.all(task_num>=tasks_per_wt):
                break

        t1 = time.time()
        
        wind_speed_error_hist = wind_speed_forecast[:(Din+run)*24,:]-wind_speed_true[:(Din+run)*24,:]
        wave_height_error_hist = wave_height_forecast[:(Din+run)*24]-wave_height_true[:(Din+run)*24]
        el_price_error_hist = el_prices_forecast[:(Din+run)*24]-el_prices_true[:(Din+run)*24]
        
        wsf_STH = wind_speed_forecast[24*(Din+run):24*(Din+run+1),:]
        whf_STH = wave_height_forecast[24*(Din+run):24*(Din+run+1)]
        wsf_LTH = wind_speed_forecast[24*(Din+run+1):24*(Din+run+N_D+1),:].reshape(-1,24,N_t).mean(1)
        whf_LTH = wave_height_forecast[24*(Din+run+1):24*(Din+run+N_D+1)].reshape(-1,24).mean(1)
        
        wst_STH = wind_speed_true[24*(Din+run):24*(Din+run+1),:]
        wht_STH = wave_height_true[24*(Din+run):24*(Din+run+1)]
        wst_LTH = wind_speed_true[24*(Din+run+1):24*(Din+run+N_D+1),:].reshape(-1,24,N_t).mean(1)
        wht_LTH = wave_height_true[24*(Din+run+1):24*(Din+run+N_D+1)].reshape(-1,24).mean(1)
        
        epf_STH = el_prices_forecast[24*(Din+run):24*(Din+run+1)]
        ept_STH = el_prices_true[24*(Din+run):24*(Din+run+1)]
        epf_LTH = el_prices_forecast[24*(Din+run+1):24*(Din+run+N_D+1)].reshape(-1,24).mean(1)
        ept_LTH = el_prices_true[24*(Din+run+1):24*(Din+run+N_D+1)].reshape(-1,24).mean(1)
        
        #prepare degradation data
        #if a task is completed, switch to next degradation profile
        if np.any(task_num!=prev_tn):
            deg_profile = deg_profiles[task_num, np.arange(N_t)]
            changed_wt = np.where(task_num!=prev_tn)[0]
            prev_tn = task_num.copy()
            for i in changed_wt:
                if N_S>1:
                    RUL_nominals_list[i] = np.array(
                        pd.read_csv(deg_data_folder+"/Failuretime_Scenarios_float/FailureT_scenarios_"+deg_profile[i]+".csv", 
                                    header=None, usecols=[j for j in range(N_S)]))
                else:
                    RUL_nominals_list[i] = np.array(
                        pd.read_csv(deg_data_folder+"/Failuretime_Scenarios_float/FailureT_scenarios_"+deg_profile[i]+".csv", 
                                    header=None)).mean(-1).reshape(-1,1)
                equivalent_time[i] = 0. #reset equivalent time
                t_0[i] = 1 #reset turbine last maintenance age
                tau[i] = repair_times[task_num[i],i].copy()
                RUL_true0[i,0] = RUL_nominals_list[i].shape[0]
                RUL_true[i,0] = RUL_true0[i,0]
        
        #NEXT: select the row of each list depending on how many steps forward we move (equivalent time transform)
        RUL_nom = np.zeros((N_t, N_S))
        for i in range(N_t):
            #if the eq time is more than the max number of cycles (counting from the beginning), 
            #then the machine fails, otherwise we use RUL scenarios from the corresponding RUL_nominals_list:
            if equivalent_time[i] <= RUL_true0[i,0]:
                RUL_nom[i,:] = RUL_nominals_list[i][int(equivalent_time[i]),:N_S]
                if (CMS) | (TBS): RUL_nom[i,0] = N_D+2 #No degradation information
        
        (norm_power, norm_powerL, el_price, el_priceL, op_time, op_timeL, 
        RUL_LG, RUL_LG_LTH, access, access_LTH ,ws_STH, wh_STH,_,_
        ) = dataLoader(wsf_STH, wsf_LTH, whf_STH, whf_LTH, epf_STH, epf_LTH, 
            wind_speed_error_hist, wave_height_error_hist, el_price_error_hist, N_S, 
            max_wind, max_wave, tau, power_curve_map, loading_map, sim_day_ahead=False, 
            NAIVE=NAIVE)
        
        
        stochastic_input.append({"Normalized power STH": norm_power.copy(),
                                 "Normalized power LTH": norm_powerL.copy(),
                                 "El. prices STH": el_price.copy(),
                                 "El. prices LTH": el_priceL.copy(),
                                 "Mission time STH": op_time.copy(),
                                 "Mission time LTH": op_timeL.copy(),
                                 "Accessibility STH": access.copy(),
                                 "Accessibility LTH": access_LTH.copy(),
                                 "RUL nominal": RUL_nom.copy(),
                                 "RUL loss/gain STH": RUL_LG.copy(),
                                 "RUL loss/gain LTH": RUL_LG_LTH.copy(),
                                 "WT age": t_0.copy(),
                                 "Continued maintenance": rho.copy()})
        
        stochastic_output.append(POSYDON(norm_power, norm_powerL, el_price, el_priceL, op_time, op_timeL, RUL_nom, RUL_LG, RUL_LG_LTH, access_LTH, t_0,
                                         rho, curtailment=0, scalars=None, STH_sol=None, return_all_vars=True,
                                         rel_gap = mip_gap, NOYAW=NOYAW, CMS=CMS, TBS=TBS))
        
        if stochastic_output[-1]==None: break #IN CASE INFEAS IS ENCOUNTERED
        
        stoch_STH_maint = stochastic_output[run]['STH_sched']
        stoch_STH_yaw = stochastic_output[run]['STH_yaw']
        sth_sched = [stoch_STH_maint, stoch_STH_yaw]

        (norm_power, norm_powerL, el_price, el_priceL, op_time, op_timeL, 
        RUL_LG, RUL_LG_LTH, access, access_LTH ,ws_STH, wh_STH,_,_
        ) = dataLoader(wst_STH, wst_LTH, wht_STH, wht_LTH, ept_STH, ept_LTH, 
            wind_speed_error_hist, wave_height_error_hist, el_price_error_hist, N_S, 
            max_wind, max_wave, tau, power_curve_map, loading_map, sim_day_ahead=True)
        

        
        true_input.append({"Normalized power STH": norm_power.copy(),
                         "Normalized power LTH": norm_powerL.copy(),
                         "El. prices STH": el_price.copy(),
                         "El. prices LTH": el_priceL.copy(),
                         "Mission time STH": op_time.copy(),
                         "Mission time LTH": op_timeL.copy(),
                         "Accessibility STH": access.copy(),
                         "Accessibility LTH": access_LTH.copy(),
                         "RUL nominal": RUL_true.copy(),
                         "RUL loss/gain STH": RUL_LG.copy(),
                         "RUL loss/gain LTH": RUL_LG_LTH.copy(),
                         "WT age": t_0.copy(),
                         "Continued maintenance": rho.copy()})
        
        true_output.append(POSYDON(norm_power, norm_powerL, el_price, el_priceL, op_time, op_timeL, RUL_true, RUL_LG, RUL_LG_LTH, access_LTH, t_0,
                                   rho, curtailment=0, scalars=None, STH_sol=sth_sched, return_all_vars=True,
                                   rel_gap = 0.0001))
        
        
        if true_output[-1]==None: break #IN CASE INFEAS IS ENCOUNTERED
        
        
        #Update the equivalent time steps given the YM actions taken in the STH,
        #to inform the failure scenarios of the next round
        equivalent_time += 1/24*np.sum(np.sum(stoch_STH_yaw * RUL_LG[:,:,:,0], -1),0)
        
        #The true RUL for the next round will be the difference between the RUL
        #in the beginning and the equivalent time steps that we are in the degradation process
        RUL_true[:,0] = RUL_true0[:,0] - equivalent_time
        RUL_true[RUL_true<0.0]=0
        
        t_0 += 1 #move forward in time by 1 day
        
        eq_times.append(equivalent_time.copy())
        
        true_STH = np.array(true_output[run]['STH_sched'])
        
        wind_speed_error_hist = np.append(wind_speed_error_hist,
                                          wsf_STH-wst_STH, 0)
        wave_height_error_hist = np.append(wave_height_error_hist,
                                          whf_STH-wht_STH, 0)
        
        
        #check for unfinished tasks, or completed tasks in the STH:
        for i in range(N_t):
            if (np.any(true_STH[:,i])>0):
                if true_output[run]['remaining_maint_hrs'][i] == 0:
                    rho[i]=0
                    task_num[i] += 1 #RUL scenarios will be updated in the next iter
                else:
                    rho[i]=1
                    tau[i]=true_output[run]['remaining_maint_hrs'][i,0]


        run+=1
        one_iteration_time.append(time.time()-t1)
        
        if run==max_iter: break
        
    return {  'stochastic_sol':stochastic_output,
              'stochastic_input': stochastic_input,
              'simulation': true_output,
              'true_input':true_input,
              'time_per_roll': np.array(one_iteration_time),
              'equivalent_times': np.array(eq_times)
              }


#%%

def makelight(sol_dict):
    for i in sol_dict.keys():
        for j in range(len(sol_dict[i]['time_per_roll'])):
            sol_dict[i]['stochastic_sol'][j]['model'] = None
            sol_dict[i]['simulation'][j]['model'] = None
    return sol_dict


