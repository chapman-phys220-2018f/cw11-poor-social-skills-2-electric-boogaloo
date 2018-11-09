#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###
# Name: Trevor Kling
# Student ID: 002270716
# Email: kling109@mail.chapman.edu
# Course: PHYS220/MATH220/CPSC220 Fall 2018
# Assignment: CW 11
###

import numpy as np

def eulerHelper(initPoint, delT):
    """
    Helper Method for Euler's method for calculating differential equations.

    Parameters:
    -----------
    initPoint: [float, float]
        The initial point u_k to be used for the approximation.  Input as a vector in order to compute multiple functions simultaneously
    delT: float > 0
        The change in time value associated with going from u_k to u_{k+1}

    Returns:
    --------
    u_{k+1}: [float, float]
        The new point approximated by the method.
    """
    J = np.matrix('0 1; -1 0')
    slopes = J @ initPoint
    return initPoint + (delT * slopes)

def euler(N, u):
    """
    Approximates the values of cos(t) and -sin(t) between 0 and 10 pi.

    Parameters:
    -----------
    N: int > 0
        The number of divisions to use for approximations.  Defines delT and the number of values in the array returned.
    u: float
        The initial value of the function to be used.

    Returns:
    --------
    eulerApprox: n by 2 array [[float,float]]
        An array of approximated function values for cos(t) and -sin(t)
    """
    tRange = np.arange(0, 10*np.pi, 2*np.pi/N)
    eulerApprox = np.zeros((len(tRange)+1, 2))
    delT = tRange[1] - tRange[0]
    eulerApprox[0] = u
    n = 0
    for t in tRange:
        n += 1
        eulerApprox[n] = eulerHelper(eulerApprox[n-1], delT)
    return eulerApprox

def heunHelper(initPoint, delT):
    """
    Helper Method for Heun's method for calculating differential equations.

    Parameters:
    -----------
    initPoint: [float, float]
        The initial point u_k to be used for the approximation.  Input as a vector in order to compute multiple functions simultaneously
    delT: float > 0
        The change in time value associated with going from u_k to u_{k+1}

    Returns:
    --------
    u_{k+1}: [float, float]
        The new point approximated by the method.
    """
    nextApprox = eulerHelper(initPoint, delT)
    J = np.matrix('0 1; -1 0')
    return initPoint + (delT / 2)*((J @ (initPoint + nextApprox).reshape((2,1))).reshape(2))

def heun(N, u):
    """
    Approximates the values of cos(t) and -sin(t) between 0 and 10 pi.

    Parameters:
    -----------
    N: int > 0
        The number of divisions to use for approximations.  Defines delT and the number of values in the array returned.
    u: float
        The initial value of the function to be used.

    Returns:
    --------
    heunApprox: n by 2 array [[float,float]]
        An array of approximated function values for cos(t) and -sin(t)
    """
    tRange = np.arange(0, 10*np.pi, 2*np.pi/N)
    heunApprox = np.zeros((len(tRange)+1, 2))
    delT = tRange[1] - tRange[0]
    heunApprox[0] = u
    n = 0
    for t in tRange:
        n += 1
        heunApprox[n] = heunHelper(heunApprox[n-1], delT)
    return heunApprox

def rungeKuttaSecondHelper(initPoint, delT):
    """
    Helper Method for the second-order Runge Kutta method for calculating differential equations.

    Parameters:
    -----------
    initPoint: [float, float]
        The initial point u_k to be used for the approximation.  Input as a vector in order to compute multiple functions simultaneously
    delT: float > 0
        The change in time value associated with going from u_k to u_{k+1}

    Returns:
    --------
    u_{k+1}: [float, float]
        The new point approximated by the method.
    """
    J = np.matrix('0 1; -1 0')
    k1 = delT*(J @ initPoint)
    k2 = delT*(J @ (initPoint + (k1 / 2)).reshape((2,1))).reshape(2)
    return initPoint + k2

def rungeKuttaSecond(N, u):
    """
    Approximates the values of cos(t) and -sin(t) between 0 and 10 pi.

    Parameters:
    -----------
    N: int > 0
        The number of divisions to use for approximations.  Defines delT and the number of values in the array returned.
    u: float
        The initial value of the function to be used.

    Returns:
    --------
    rksa: n by 2 array [[float,float]]
        An array of approximated function values for cos(t) and -sin(t)
    """
    tRange = np.arange(0, 10*np.pi, 2*np.pi/N)
    rksa = np.zeros((len(tRange)+1, 2))
    delT = tRange[1] - tRange[0]
    rksa[0] = u
    n = 0
    for t in tRange:
        n+=1
        rksa[n] = rungeKuttaSecondHelper(rksa[n-1], delT)
    return rksa

def rungeKuttaFourthHelper(initPoint, delT):
    """
    Helper Method for the fourth-order Runge Kutta method for calculating differential equations.

    Parameters:
    -----------
    initPoint: [float, float]
        The initial point u_k to be used for the approximation.  Input as a vector in order to compute multiple functions simultaneously
    delT: float > 0
        The change in time value associated with going from u_k to u_{k+1}

    Returns:
    --------
    u_{k+1}: [float, float]
        The new point approximated by the method.
    """
    J = np.matrix('0 1; -1 0')
    k1 = delT*(J @ initPoint)
    k2 = delT*(J @ (initPoint + (k1 / 2)).reshape((2,1))).reshape(2)
    k3 = delT*(J @ (initPoint + (k2 / 2)).reshape((2,1))).reshape(2)
    k4 = delT*(J @ (initPoint+k3).reshape((2,1))).reshape(2)
    return initPoint + (k1 + 2*k2 + 2*k3 + k4)/6

def rungeKuttaFourth(N, u):
    """
    Approximates the values of cos(t) and -sin(t) between 0 and 10 pi.

    Parameters:
    -----------
    N: int > 0
        The number of divisions to use for approximations.  Defines delT and the number of values in the array returned.
    u: float
        The initial value of the function to be used.

    Returns:
    --------
    rkfa: n by 2 array [[float,float]]
        An array of approximated function values for cos(t) and -sin(t)
    """
    tRange = np.arange(0, 10*np.pi, 2*np.pi/N)
    rkfa = np.zeros((len(tRange)+1, 2))
    delT = tRange[1] - tRange[0]
    rkfa[0] = u
    n = 0
    for t in tRange:
        n+=1
        rkfa[n] = rungeKuttaFourthHelper(rkfa[n-1], delT)
    return rkfa

