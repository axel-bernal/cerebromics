"""
Translated from java.

See http://www.cs.rit.edu/~ncs/color/t_convert.html for the source of these magic numbers.
See also http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html. These
equations use the sRGB standard from that page.
These transforms depend on the choice of a white point - we use D65, which I believe is 
standard for representing daylight colors in photography.
Lab is a color space that attempts to "linearize the perceptibility of color differences",
so that a Euclidean distance in Lab space is a perceptually reasonable distance
between colors.
"""

import numpy as np
import math

xtor    =  np.array([[3.240479, -1.537150, -0.498535], [-0.969256, 1.875992, 0.041556], [0.055648, -0.204043, 1.057311]])
rtox    =  np.array([[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169], [0.019334, 0.119193, 0.950227]])
    
# Tristimulus values for reference D65 white.
xn = 0.95047
yn = 1.0000
zn = 1.08883
T  = 0.008856
T2 = 0.206893

def RgbToLab (rgb):
    
    xyz = RgbToXyz(rgb)
    return XyzToLab(xyz)

def LabToRgb (lab):
      
    xyz = LabToXyz(lab)
    rgb = XyzToRgb(xyz)
    return rgb

def RgbToXyz(rgb): 
    return matMult(rgb, rtox);

def XyzToLab(xyz):    
    xr  = xyz[0] / xn
    yr  = xyz[1] / yn
    zr  = xyz[2] / zn
    
    lab = np.zeros(3)
    if (yr > T):
        lab[0]  = (116 * math.pow(yr, 0.3333333) - 16)
    else:
        lab[0]  = (yr * 903.3)
    
    lab[1]  = (500 * (fl(xr) - fl(yr)))
    lab[2]  = (200 * (fl(yr) - fl(zr)))
    return lab
  
def fl (r):
    
    if (r > T):
        return math.pow(r, 0.333333)
    return 7.787 * r + 16.0 / 116
 
# Only handles a subset of the feature space - is it enough?
def LabToXyz(lab):

    xyz = np.zeros(3)
    
    # Compute y
    P    = (lab[0] + 16) / 116
    P3   = math.pow(P, 3)
    if (P3 > T):
        xyz[1]  = P3
        fy      = P
    else:
        xyz[1]  = (lab[0] / 903.3)
        fy      = (7.787 * xyz[1] + 16.0/116)
    
    fx  = (lab[1] / 500 + fy)
    if (fx > T2):
        xyz[0]  = math.pow(fx, 3)
    else:
        xyz[0]  = ((fx - 16.0 / 116) / 7.787)
    
    fz  = fy - lab[2] / 200
    if (fz > T2):
        xyz[2]  = math.pow(fz, 3)
    else:
        xyz[2]  = ((fz - 16.0 / 116) / 7.787)
    
    # White point adjustment
    xyz[0] = (xyz[0] * 0.950456)
    xyz[2] = (xyz[2] * 1.088754)
    
    xyz[0]      = (xn * math.pow(P + lab[1]/500, 3))
    xyz[1]      = (yn * math.pow(P, 3))
    xyz[2]      = (zn * math.pow(P - lab[2] / 200, 3))
    
    return xyz

def XyzToRgb(xyz):   
    return matMult(xyz, xtor)

def matMult(rgb, rtox):   
    return np.dot(rtox, rgb)

if __name__ == "__main__":
    lab = np.array([39.5458735132, 5.14864166633,2.96902171616])
    rgb = np.array([0.13745728,0.10267905,0.09879114])
  
    lab2 = RgbToLab(rgb)
    rgb2 = LabToRgb(lab)
    print lab2
    print rgb2

