import numpy as np
import math



def cartesian_to_frenet(rs, rx, ry, rtheta, rkappa, rdkappa, x, y, v, a, theta, kappa):
    dx = x - rx
    dy = y - ry

    cos_theta_r = math.cos(rtheta)
    sin_theta_r = math.sin(rtheta)

    cross_rd_nd = cos_theta_r * dy - sin_theta_r * dx
    d_condition = np.zeros(3)
    s_condition = np.zeros(3)

    d_condition[0] = math.copysign(math.sqrt(dx * dx + dy * dy), cross_rd_nd)

    delta_theta = theta - rtheta
    tan_delta_theta = math.tan(delta_theta)
    cos_delta_theta = math.cos(delta_theta)

    one_minus_kappa_r_d = 1 - rkappa * d_condition[0]
    d_condition[1] = one_minus_kappa_r_d * tan_delta_theta

    kappa_r_d_prime = rdkappa * d_condition[0] + rkappa * d_condition[1]

    d_condition[2] = -kappa_r_d_prime * tan_delta_theta + \
                     one_minus_kappa_r_d / (cos_delta_theta ** 2) * \
                     (kappa * one_minus_kappa_r_d / cos_delta_theta - rkappa)

    s_condition[0] = rs

    s_condition[1] = v * cos_delta_theta / one_minus_kappa_r_d

    delta_theta_prime = one_minus_kappa_r_d / cos_delta_theta * kappa - rkappa
    s_condition[2] = (a * cos_delta_theta - 
                      s_condition[1] ** 2 * 
                      (d_condition[1] * delta_theta_prime - kappa_r_d_prime)) / \
                      one_minus_kappa_r_d

    return s_condition, d_condition

def find_closest_index(rx, ry, x, y):
    #rx, ry are lists of points on the reference line
    #this will fin the index of the point that is closest to the cartesian coordinate (x,y)
    if len(rx)!=len(ry):
        raise Exception("rx and ry must be the same length")
    
    index = None
    distance = float("inf")
    for i in range(0,len(rx)):
        if math.hypot(rx[i]-x,ry[i]-y) < distance:
            index = i
            distance = math.hypot(rx[i]-x,ry[i]-y)
    return index

def normalize_angle(angle):
    #Normalize the angle to the range [-pi, pi]
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle