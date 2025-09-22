#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 09:51:37 2025

@author: yycloud
"""

# app.py
from flask import Flask, render_template, request,make_response
from flask import send_from_directory
import contrast_calculator
import os

app = Flask(__name__,
            static_folder='static',
            template_folder='templates')
            
    
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 获取用户输入
        planet_params = {
            'a': request.form.get('a'),
            'e': request.form.get('e'),
            'per': request.form.get('per'),
            'tp': request.form.get('tp'),
            'I_deg': request.form.get('I_deg'),
            'omega_deg': request.form.get('omega_deg'),
            'M': request.form.get('M'),
            'R_p': request.form.get('R_p')
        }
        
        star_params = {
            'M_s': request.form.get('M_s'),
            'R_s': request.form.get('R_s'),
            'T': request.form.get('T')
        }
        
        wavelength = request.form.get('wavelength')
        distance = request.form.get('distance')
        
        # 计算对比度
        try:
            data = contrast_calculator.calculate_contrast(planet_params, star_params, wavelength, distance)
            return render_template('index.html', 
                                 # images=data['images'],
                                  results=data['results'],
                                  planet_params=planet_params,
                                  star_params=star_params,
                                  wavelength=wavelength,
                                  distance=distance)
        except Exception as e:
            return render_template('index.html', error=str(e))
    
    # 默认值
    default_planet = {
        'a': 1, 'e': 0.5, 'per': 1000, 'tp': 0.0,
        'I_deg': 90.0, 'omega_deg': 30.0, 'M': 1, 'R_p': 1
    }
    default_star = {'M_s': 1.0, 'R_s': 1.0, 'T': 6000.0}
    default_wavelength = 3
    default_distance = 30
    
    return render_template('index.html',
                          planet_params=default_planet,
                          star_params=default_star,
                          wavelength=default_wavelength,
                          distance=default_distance)


if __name__ == '__main__':
    # 启用调试模式，自动重新加载更改
    app.run(debug=True, host='0.0.0.0', port=5001)
