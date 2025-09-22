#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 09:43:11 2025

@author: yycloud
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
#import corner
from matplotlib import rcParams, gridspec
from matplotlib.ticker import MaxNLocator
#from astropy.time import Time
import copy
import random
import glob
#import seaborn as sns
from configparser import ConfigParser
from math import sin, cos, tan, sqrt
#from numpy import sin, cos, tan
from scipy.stats import multivariate_normal
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.ticker import AutoMinorLocator
#from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d
from scipy import stats, signal
#from ptemcee import Sampler as PTSampler
import time
import re
import os
import scipy
#from corgietc.corgietc import corgietc
import json
import EXOSIMS.Prototypes.TargetList
import EXOSIMS.Prototypes.TimeKeeping
import EXOSIMS.Observatory.ObservatoryL2Halo
import EXOSIMS.PlanetPhysicalModel.ForecasterMod
from EXOSIMS.util.deltaMag import deltaMag
from EXOSIMS.util.phaseFunctions import phi_lambert
from keplertools import fun as kepler_fun
import astropy.units as u
from astropy.time import Time
from scipy.integrate import quad
from astropy import constants as const
from io import BytesIO
import base64
# 函数：计算轨道投影长度s并画图


# 先复用你提供的 kepler 和 true_anomaly 函数
def kepler(Marr, eccarr):
    conv = 1.0e-12
    k = 0.85
    Earr = Marr + np.sign(np.sin(Marr)) * k * eccarr
    fiarr = Earr - eccarr * np.sin(Earr) - Marr
    convd = np.where(np.abs(fiarr) > conv)[0]
    nd = len(convd)

    while nd > 0:
        M = Marr[convd]
        ecc = eccarr[convd]
        E = Earr[convd]

        fi = fiarr[convd]
        fip = 1 - ecc * np.cos(E)
        fipp = ecc * np.sin(E)
        fippp = 1 - fip

        d1 = -fi / fip
        d2 = -fi / (fip + d1 * fipp / 2.0)
        d3 = -fi / (fip + d2 * fipp / 2.0 + d2**2 * fippp / 6.0)
        E = E + d3
        Earr[convd] = E
        fiarr = Earr - eccarr * np.sin(Earr) - Marr
        convd = np.where(np.abs(fiarr) > conv)[0]
        nd = len(convd)

    return Earr

def true_anomaly(t, tp, per, e):
    m = 2 * np.pi * (((t - tp) / per) - np.floor((t - tp) / per))
    eccarr = np.zeros_like(t) + e
    E = kepler(m, eccarr)
    n1 = 1 + e
    n2 = 1 - e
    nu = 2 * np.arctan(np.sqrt(n1/n2) * np.tan(E/2))
    return nu

def planck_lambda(wavelength, T):
    """
    黑体辐射函数 B_lambda(T)

    ----------
    B_lambda : Quantity
        光谱辐射亮度 [W / (m2 sr m)]
    """
    # 保证输入有单位
    wavelength = wavelength* u.um.to(u.m)
    T = T * u.K if not hasattr(T, "unit") else T

    # 普朗克公式

    x = ((const.h * const.c) / (wavelength * const.k_B * T)).decompose().value  # 无量纲
    exp_x = np.exp(np.clip(x, None, 700))

    B_lambda = (2 * const.h * const.c**2 / wavelength**5) / (exp_x - 1)
   
  

    return B_lambda



def orbit_projection(a, e, per, tp, I_deg, omega_deg, t_obs=None,num_points=None):
    """
    直接从 t 均匀采样整个轨道，计算每个 t 对应的投影长度 s，并画图。

    Args:
        a: 半长轴 [AU]
        e: 偏心率
        per: 轨道周期
        tp: 近日点时间
        I_deg: 倾角 [deg]
        omega_deg: 近日点角 [deg]
        num_points: 轨道采样点数

    Returns:
        t_arr: 时间数组
        s_arr: 投影长度数组
    """
    if t_obs is None:
        if num_points is None:
            num_points = 1000
        t_arr = np.linspace(tp, tp + per, num_points)
    else:
        t_arr = np.atleast_1d(t_obs)
    # 弧度转换
    I = np.radians(I_deg)
    omega = np.radians(omega_deg)

    # 均匀采样时间 t
    #t_arr = np.linspace(tp, tp + per, num_points)

    # 对每个 t 计算真近点角 nu
    nu_arr = true_anomaly(t_arr, tp, per, e)

    # 半径 r
    r_arr = a * (1 - e**2) / (1 + e * np.cos(nu_arr))

    # 投影长度 s
    s_arr = r_arr * np.sqrt(1 - (np.sin(I)**2) * (np.sin(nu_arr + omega)**2))

    
    return t_arr,r_arr,s_arr

def theta_filter(s_arr,distance):
    theta_arr = s_arr / distance * 1000  
    return theta_arr

def empirical_albedo(M, per, wavelength):
    """
    根据行星质量 M[Mj]、周期 per[days]和波长 lambda[um]估计反照率 Ag。
    
    Args:
        M: 行星质量 [Mj]
        per: 轨道周期 [days]
        wavelength: 波长 [um]
    
    Returns:
        Ag: 经验反照率
    """
    # 波段判断
    wavelength = np.atleast_1d(wavelength)  # 保证是数组
    Ag = np.zeros_like(wavelength, dtype=float)
    
    # ---------------- 行星类型判断 ----------------
    if M < 0.03:   # 小于 10 M⊕
        planet_type = 'earth'
    elif M < 0.15: # ~50 M⊕，海王星型
        planet_type = 'neptune'
    elif M < 0.4:  # 土星型
        planet_type = 'saturn'
    else:          # >=0.4 Mj
        if per < 10:
            planet_type = 'hot_jupiter'
        else:
            planet_type = 'cold_jupiter'
    
    # ---------------- 不同类型的反照率表 ----------------
    Ag_table = {
        'earth':       {'visible': 0.3,  'nir': 0.2,  'mir': 0.001},
        'hot_jupiter': {'visible': 0.05, 'nir': 0.02, 'mir': 0.001},
        'cold_jupiter':{'visible': 0.5,  'nir': 0.4,  'mir': 0.001},
        'saturn':      {'visible': 0.4,  'nir': 0.3,  'mir': 0.001},
        'neptune':     {'visible': 0.3,  'nir': 0.25, 'mir': 0.001}
    }
    
    # ---------------- 波段分类 ----------------
    visible_mask = wavelength < 0.7
    nir_mask = (wavelength >= 0.7) & (wavelength < 2.0)
    mir_mask = wavelength >= 2.0
    
    Ag[visible_mask] = Ag_table[planet_type]['visible']
    Ag[nir_mask]     = Ag_table[planet_type]['nir']
    Ag[mir_mask]     = Ag_table[planet_type]['mir']
    
    # 如果输入是标量，就返回标量
    if Ag.size == 1:
        return Ag.item()
    return Ag

def compute_bond_albedo(wavelength_all, Ag, T_star):
    """
    计算 Bond albedo
    ----------
    wavelength : array
        波长 (单位: 米)
    Ag_lambda : array
        对应波长的几何反照率 A_g(λ)
    T_star : float
        恒星温度 (K)
    """
    if not hasattr(wavelength_all, "unit"):
        wl = (wavelength_all * u.um).to(u.m).value  # 默认输入单位是微米
    else:
        wl = wavelength_all.to(u.m).value
    # 恒星光谱 (黑体近似)，权重
    F_lambda = planck_lambda(wl, T_star)

    # 积分
    numerator = np.trapz(Ag * F_lambda, wl)
    denominator = np.trapz(F_lambda, wl)

    Ab = numerator / denominator
    return Ab

def phase_function(e,I_deg, omega_deg, per, tp, t_obs=None,num_points=None):
    """
    计算行星的相位函数 Phi(beta)

    Args:
        I_deg: 轨道倾角 [度]
        omega_deg: 近日点角 [度]
        nu_rad: 真近点角 [弧度]

    Returns:
        beta: 相位角 [弧度]
        Phi: 相位函数 Phi(beta)
    """
    if t_obs is None:
        if num_points is None:
            num_points = 1000
        t_arr = np.linspace(tp, tp + per, num_points)
    else:
        t_arr = np.atleast_1d(t_obs)
    # 转弧度
    I = np.radians(I_deg)
    omega = np.radians(omega_deg)
    # 均匀采样时间 t
    #t_arr = np.linspace(tp, tp + per, num_points)

    # 对每个 t 计算真近点角 nu
    nu_arr = true_anomaly(t_arr, tp, per, e)
    # 计算相位角 beta
    cos_beta = -np.sin(I) * np.sin(omega + nu_arr)
    # 避免数值误差导致 cos_beta 超过 [-1,1]
    cos_beta = np.clip(cos_beta, -1.0, 1.0)
    beta = np.arccos(cos_beta)

    # 相位函数 Phi(beta)
    Phi = (np.sin(beta) + (np.pi - beta) * np.cos(beta)) / np.pi

    return beta, Phi


def Radius_planet(M,Rp):
    ppmod = EXOSIMS.PlanetPhysicalModel.ForecasterMod.ForecasterMod()
    
    M_val = M * u.M_jup
    Rp_val= Rp * u.jupiterRad
    if Rp_val>0:
        Rp_val = Rp_val.to(u.jupiterRad).value 
    else:
        Rp_val = ppmod.calc_radius_from_mass(M_val)[0].to(u.jupiterRad).value
    return Rp_val

#def compute_q(Phi,beta):
#    """计算相位积分因子 q"""
#    integrand = lambda beta: Phi * np.sin(beta)
 #   q_val, _ = 2 * quad(integrand, 0, np.pi)
 #   return q_val

def equilibrium_temperature(T_star, R_star, a, Ab, f=1.0):
    """
    计算行星平衡温度 (使用 Ag → Ab)

    参数
    ----------
    T_star : float or Quantity
        恒星有效温度 [K]
    R_star : float or Quantity
        恒星半径 [Rsun 或 AU]
    a : float or Quantity
        半长轴 [AU]
    Ag : float
        几何反照率
    f : float
        再分布因子 (默认=1)

    返回
    ----------
    Teq : Quantity
        平衡温度 [K]
    """
    # 保证单位
    T_star = T_star * u.K if not hasattr(T_star, "unit") else T_star
    R_star = R_star * u.R_sun if not hasattr(R_star, "unit") else R_star
    a = a * u.AU if not hasattr(a, "unit") else a

    # 相位积分因子
    #q = compute_q(Phi,beta)
    #q=3/2
    # Bond 反照率
    #Ab = q * Ag

    # 转换 R_star 为 AU
    R_star_au = R_star.to(u.AU)

    # 平衡温度公式
    Teq = T_star * np.sqrt(R_star_au / (2.0 * a)) * (1 - Ab) ** 0.25 * f ** 0.25 * u.K
    return Teq

def thermal_contrast(wavelength, Rp, R_star, Tp,T_star):
    """
    计算行星的热辐射对比度 C_thermal(lambda)
    ----------
    wavelength : Quantity
    Rp : Quantity
    R_star : Quantity
        恒星半径 [Rsun, m 等]
    Tp : float or Quantity
        行星温度 [K]
    T_star : float or Quantity
        恒星温度 [K]

    返回
    ----------
    C_th : ndarray
        热辐射对比度
    """
    Rp = (Rp * u.jupiterRad).to(u.m)
    R_star = (R_star * u.R_sun).to(u.m)
    Tp = Tp * u.K if not hasattr(Tp, "unit") else Tp
    T_star = T_star * u.K if not hasattr(T_star, "unit") else T_star
    wavelength*u.um.to(u.m)
    ratio_area = (Rp / R_star)**2
    #ratio_area=1
    Bp = planck_lambda(wavelength, Tp)
    Bs = planck_lambda(wavelength, T_star)
    C_theraml=(ratio_area * (Bp / Bs)).decompose()

    return C_theraml

# contrast_calculator.py
class Planet:
    def __init__(self, a, e, per, tp, I_deg, omega_deg, M,R_p):
        self.a = a
        self.e = e
        self.per = per
        self.tp = tp
        self.I_deg = I_deg
        self.omega_deg = omega_deg
        self.M = M
        self.Rp=R_p
class Star:
    def __init__(self, M_s,R_s,T):
        self.M_s=M_s
        self.R_s=R_s
        self.T=T

def calculate_contrast(planet_params, star_params, wavelength, distance):
    """
    计算对比度的主函数
    返回：三个图的Base64编码图像和数值结果
    """
    # 从参数中提取值
    a = float(planet_params['a'])
    e = float(planet_params['e'])
    per = float(planet_params['per'])
    tp = float(planet_params['tp'])
    I_deg = float(planet_params['I_deg'])
    omega_deg = float(planet_params['omega_deg'])
    M = float(planet_params['M'])
    R_p = float(planet_params['R_p'])
    
    M_s = float(star_params['M_s'])
    R_s = float(star_params['R_s'])
    T = float(star_params['T'])
    
    wavelength = float(wavelength)
    distance = float(distance)
    
    # 创建行星和恒星对象
    planet = Planet(a, e, per, tp, I_deg, omega_deg, M, R_p)
    star = Star(M_s, R_s, T)
    
    # 执行你的计算代码...
    # ================== Orbit & Phase ==================
    t_arr, r_arr, s_arr = orbit_projection(
        planet.a, planet.e, planet.per, planet.tp, planet.I_deg, planet.omega_deg, t_obs=None
    )
    beta, Phi = phase_function(
        planet.e, planet.per, planet.tp, planet.I_deg, planet.omega_deg, t_obs=None
    )
    angular=s_arr/distance
    # ================== Radius & Albedo ==================
    wavelength_all = np.linspace(0.05, 25, 1000) * u.um
    Ag = empirical_albedo(planet.M, planet.per, wavelength)
    Ab = compute_bond_albedo(wavelength_all, Ag, star.T)
    Rp_val = Radius_planet(planet.M, planet.Rp)*u.jupiterRad.to(u.AU)
    Rp_j = Radius_planet(planet.M, planet.Rp)
    
    # ================== Temperature & Thermal Contrast ==================
    Tp = equilibrium_temperature(star.T, star.R_s, planet.a, Ab, f=1.0)
    C_thermal_arr = thermal_contrast(wavelength, Rp_j, star.R_s, Tp, star.T)
    
    # ================== Reflection Contrast ==================
    C_refl = Ag * Phi * (Rp_val/ r_arr)**2
    
    # ================== Total Contrast ==================
    C_total = C_refl + C_thermal_arr
    
    # 计算平均值和范围
    results = {
        'thermal_mean': np.mean(C_thermal_arr),
        'thermal_min': np.min(C_thermal_arr),
        'thermal_max': np.max(C_thermal_arr),
        'refl_mean': np.mean(C_refl),
        'refl_min': np.min(C_refl),
        'refl_max': np.max(C_refl),
        'total_mean': np.mean(C_total),
        'total_min': np.min(C_total),
        'total_max': np.max(C_total),
        'albedo': Ag,
        'temperature': Tp.value if hasattr(Tp, 'value') else Tp,
        'radius': Rp_j,
        'separation':np.mean(s_arr),
        'angular':np.mean(angular)*1000
    }
    
    # 创建三个图表
    #images = []
    
    # 图1: contrast vs time
#    plt.figure(figsize=(8,4))
#    plt.plot(t_arr, C_total - C_refl, '-', color='dodgerblue', label="Thermal")
#    plt.plot(t_arr, C_refl, '-', color='orange', label="Reflection")
#    plt.plot(t_arr, C_total, '-', color='green', label="Total")
#    plt.xlabel('Time')
#    plt.ylabel('Contrast')
#    plt.yscale('log')
#    plt.title('Contrast vs Time')
#    plt.legend()
#    plt.grid(True)
    
    # 将图表转换为Base64编码
#    img_buffer = BytesIO()
#    plt.savefig(img_buffer, format='png')
#    img_buffer.seek(0)
#    img_str1 = base64.b64encode(img_buffer.getvalue()).decode()
#    images.append(img_str1)
#    plt.close()
    
    # 图2: contrast vs inclination
#    I_list = np.linspace(0, 180, 50)
#    C_inc = []
#
#    for I_deg_val in I_list:
#        _, r_arr_tmp, _ = orbit_projection(planet.a, planet.e, planet.per, planet.tp, I_deg_val, planet.omega_deg, t_obs=3.5)
#        _, Phi_tmp = phase_function(planet.e, I_deg_val, planet.omega_deg, planet.per, planet.tp, t_obs=3.5)
#       
#        C_refl_tmp = Ag * Phi_tmp * (Rp_val/ r_arr_tmp)**2
#        
#        C_total_tmp = np.mean(C_refl_tmp + C_thermal_arr)  # 取平均对比度
#        C_inc.append(C_total_tmp)
#
#    plt.figure(figsize=(8,4))
#    plt.plot(I_list, C_inc, '-', color='purple')
#    plt.xlabel("Inclination [deg]")
#    plt.ylabel("Mean Contrast")
#    plt.yscale("log")
#    plt.title("Contrast vs Inclination")
#    plt.grid(True)
#    
#    img_buffer = BytesIO()
#    plt.savefig(img_buffer, format='png')
#    img_buffer.seek(0)
#    img_str2 = base64.b64encode(img_buffer.getvalue()).decode()
#    images.append(img_str2)
#    plt.close()
#    
#    # 图3: contrast vs wavelength
#    bands = {
#        "B": (0.40, 0.50),
#        "V": (0.50, 0.60),
#        "R": (0.60, 0.70),
#        "I": (0.70, 0.90),
#        "Y": (0.97, 1.07),
#        "J": (1.1, 1.4),
#        "H": (1.5, 1.8),
#        "K": (2.0, 2.4),
#        "L": (3.0, 4.0),
#        "M": (4.5, 5.0),
#        "N": (8, 13)
#    }
#    wavelengths = np.linspace(0.1, 25, 1000)
#    Ag = empirical_albedo(planet.M, planet.per, wavelengths)
#    C_refl = Ag * Phi * (Rp_val/ r_arr)**2
#
#    C_thermal_arr = thermal_contrast(wavelengths, Rp_j, star.R_s, Tp, star.T)
#    C_total = C_thermal_arr + C_refl
#    
#    colors = plt.cm.tab10.colors  # 10种主色调
#    linestyles = {"total": "-", "thermal": "--", "reflection": ":"}
#
#    plt.figure(figsize=(10,6))
#    for i, (band, (lam_min, lam_max)) in enumerate(bands.items()):
#        idx = np.where((wavelengths >= lam_min) & (wavelengths <= lam_max))
#        for j, (ctype, arr) in enumerate({
#            "total": C_total,
#            "thermal": C_thermal_arr,
#            "reflection": C_refl
#        }.items()):
#            plt.plot(
#                wavelengths[idx], arr[idx],
#                label=f"{band} {ctype}" if i == 0 else "",  # 避免重复 legend
#                color=colors[i % len(colors)],
#                linestyle=linestyles[ctype]
#            )
#
#    plt.xlabel("Wavelength [µm]")
#    plt.ylabel("Contrast")
#    plt.yscale('log')
#    plt.title("Contrast vs Wavelength")
#
#    # Legend 只显示一次类型说明
#    plt.legend(ncol=3, loc='best', frameon=True, framealpha=0.8, edgecolor='gray', fontsize=9, title="Band & Component")
#
#    plt.grid(True, alpha=0.3)
#    
#    img_buffer = BytesIO()
#    plt.savefig(img_buffer, format='png')
#    img_buffer.seek(0)
#    img_str3 = base64.b64encode(img_buffer.getvalue()).decode()
#    images.append(img_str3)
#    plt.close()
    
    return {'results': results}
