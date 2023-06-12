#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 17:31:37 2022

@author: xiao

This script calculates the yield strength from 7 points on a s-s curve
input needs to be a list of stresses list at [0.25, ..., 1.5]
returns a list of moduli and yield strengths
"""
import numpy as np
from scipy import interpolate

def yield_strength_cal(target):

    strain = np.linspace(0.0025,0.015,6)
    interp_factor = 10 # density of interpolation. larger the finer
    strain_new = np.linspace(0.0025,0.015,6*interp_factor)
    
    yield_strengths = []
    moduli = []
    
    for i in range(0,len(target)):
        stress = target[i]
        E = stress[0]/0.0025 # calculate Young's modulus
        intercept = -E*0.002 # calculate the intercept of the 0.2% offsetline
        model_interp = interpolate.interp1d(strain,stress,'linear') # fit the current interpolation model
        stress_new = model_interp(strain_new) # interpolate the stress
        offsetline = E*strain_new+intercept # assemble the offset line
        
        idx_yield = np.argwhere(np.diff(np.sign(stress_new - offsetline))).flatten() # find the index of the yield stress
        yield_strength = stress_new[idx_yield][0] # '0' is used to convert array to float number
    
        yield_strengths.append(yield_strength)
        moduli.append(E)
        
    return moduli,yield_strengths