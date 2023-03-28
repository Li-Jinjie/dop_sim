#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Author: LI Jinjie
File: def_mul_states.py
Date: 2023/3/27 下午5:39
Description:
"""

# "player", "group", "HP",  # 0,1,2 base parameters
# # -------- dynamics related --------
# "east", "north", "up", "phi", "theta", "psi",  # 3, 4, 5, 6, 7, 8
# "ew", "ex", "ey", "ez",  # 9, 10, 11, 12 四元数
# "vx", "vy", "vz", "u", "v", "w",  # 13, 14, 15, 16, 17, 18 分别是世界系下的和机体系下的速度，具体选哪个和agent动力学有关
# "p", "q", "r",  # 19, 20, 21 角速度
# "Va",  # 22 airspeed in meters/sec
# "Vg",  # 23 ground speed in meters/sec
# "alpha",  # 24 angle of attack in radians
# "beta",  # 25 sideslip angle in radians
# "gamma",  # 26 flight path angle in radians
# "chi",  # 27 course angle in radians
# # 包括 steady 和 gust 的总和
# "wn",  # 28 inertial wind speed in north direction in meters/sec
# "we",  # 29 inertial wind speed in east direction in meters/sec
# "wd",  # 30 inertial wind speed in down direction in meters/sec
# "cruising_speed",  # 31 nominal speed
# # ------- delta related ------
# "o1"  # 31 rotor speed for rotor 1 in kRPM
# "o2"  # 32 rotor speed for rotor 2 in kRPM
# "o3"  # 33 rotor speed for rotor 3 in kRPM
# "o4"  # 34 rotor speed for rotor 4 in kRPM
