"""
Created on Mon Jun 24 17:13:42 2013

@author: tisimst
"""

import math

from soerp import Chi2, Exp, Gamma, N, umath, uv


print("UNCERTAIN DISTRIBUTION TEST FUNCTIONS USING GIVEN MOMENTS")
print("*" * 80)
print("Example of a three part assembly")
x1 = uv([24, 1, 0, 3, 0, 15, 0, 105])  # normally distributed
x2 = uv([37, 16, 0, 3, 0, 15, 0, 105])  # normally distributed
x3 = uv([0.5, 0.25, 2, 9, 44, 265, 1854, 14833])  # exponentially distributed

Z = (x1 * x2**2) / (15 * (1.5 + x3))
print("Results should be about:")
print(" > Mean...................  1176.45")
print(" > Variance...............  99699.682")
print(" > Skewness Coefficient...  0.70801305")
print(" > Kurtosis Coefficient...  6.1632855")
print(Z)

print("*" * 80)
print("Example of volumetric gas flow through orifice meter")
H = uv([64, 0.25, 0, 3, 0, 15, 0, 105])  # normally distributed
M = uv([16, 0.01, 0, 3, 0, 15, 0, 105])  # normally distributed
P = uv([361, 4, 0, 3, 0, 15, 0, 105])  # normally distributed
t = uv([165, 0.25, 0, 3, 0, 15, 0, 105])  # normally distributed
C = 38.4
Q = C * umath.sqrt((520 * H * P) / (M * (t + 460)))
print("Results should be about:")
print(" > Mean...................  1330.9997")
print(" > Variance...............  58.210763")
print(" > Skewness Coefficient...  0.010942207")
print(" > Kurtosis Coefficient...  3.0003269")
print(Q)

print("*" * 80)
print("Example of manufacturing tolerance stackup")
x = uv([1.5, 0.25, 2 / 3.0, 11 / 3.0, 0, 0, 0, 0])  # gamma distributed
y = uv([1.5, 0.25, 2 / 3.0, 11 / 3.0, 0, 0, 0, 0])  # gamma distributed
z = uv([1.5, 0.25, 2 / 3.0, 11 / 3.0, 0, 0, 0, 0])  # gamma distributed
w = x + y + z
print("Results should be about:")
print(" > Mean...................  4.5")
print(" > Variance...............  0.75")
print(" > Skewness Coefficient...  0.385")
print(" > Kurtosis Coefficient...  3.22")
print(w)

print("*" * 80)
print("Example of scheduling facilities (six stations)")
s1 = uv([10, 1, 0, 3, 0, 0, 0, 0])  # normal distributed
s2 = uv([20, 2, 0, 3, 0, 0, 0, 0])  # normal distributed
s3 = uv([1.5, 0.25, 0.67, 3.67, 0, 0, 0, 0])  # gamma distributed
s4 = uv([10, 10, 0.63, 3.6, 0, 0, 0, 0])  # gamma distributed
s5 = uv([0.2, 0.04, 2, 9, 0, 0, 0, 0])  # exponental distributed
s6 = uv([10, 20, 0.89, 4.2, 0, 0, 0, 0])  # chi-square distributed
T = s1 + s2 + s3 + s4 + s5 + s6
print("Results should be about:")
print(" > Mean...................  51.7")
print(" > Variance...............  33.3")
print(" > Skewness Coefficient...  0.52")
print(" > Kurtosis Coefficient...  3.49")
print(T)

###############################################################################

try:
    import scipy.stats as ss
except ImportError:
    pass
else:
    print("*" * 80)
    print("SAME TEST FUNCTIONS USING DERIVED MOMENTS FROM SCIPY DISTRIBUTIONS")

    print("*" * 80)
    print("Example of a three part assembly")
    x1 = uv(rv=ss.norm(loc=24, scale=1))  # normally distributed
    x2 = uv(rv=ss.norm(loc=37, scale=4))  # normally distributed
    x3 = uv(rv=ss.expon(scale=1 / 2.0))  # exponentially distributed
    Z = (x1 * x2**2) / (15 * (1.5 + x3))
    print("Results should be about:")
    print(" > Mean...................  1176.45")
    print(" > Variance...............  99699.682")
    print(" > Skewness Coefficient...  0.70801305")
    print(" > Kurtosis Coefficient...  6.1632855")
    print(Z)

    print("*" * 80)
    print("Example of volumetric gas flow through orifice meter")
    H = uv(rv=ss.norm(loc=64, scale=0.5))
    M = uv(rv=ss.norm(loc=16, scale=0.1))
    P = uv(rv=ss.norm(loc=361, scale=2))
    t = uv(rv=ss.norm(loc=165, scale=0.5))
    C = 38.4
    Q = C * umath.sqrt((520 * H * P) / (M * (t + 460)))
    print("Results should be about:")
    print(" > Mean...................  1330.9997")
    print(" > Variance...............  58.210763")
    print(" > Skewness Coefficient...  0.010942207")
    print(" > Kurtosis Coefficient...  3.0003269")
    print(Q)

    print("*" * 80)
    print("Example of manufacturing tolerance stackup")
    # for a gamma distribution we need the following conversions:
    # scale = var/mean
    # shape = mean**2/var
    mn = 1.5
    vr = 0.25
    scale = vr / mn
    shape = mn**2 / vr
    x = uv(rv=ss.gamma(shape, scale=scale))
    y = uv(rv=ss.gamma(shape, scale=scale))
    z = uv(rv=ss.gamma(shape, scale=scale))
    w = x + y + z
    print("Results should be about:")
    print(" > Mean...................  4.5")
    print(" > Variance...............  0.75")
    print(" > Skewness Coefficient...  0.385")
    print(" > Kurtosis Coefficient...  3.22")
    print(w)

    print("*" * 80)
    print("Example of scheduling facilities (six stations)")
    s1 = uv(rv=ss.norm(loc=10, scale=1))
    s2 = uv(rv=ss.norm(loc=20, scale=2**0.5))
    mn1 = 1.5
    vr1 = 0.25
    scale1 = vr1 / mn1
    shape1 = mn1**2 / vr1
    s3 = uv(rv=ss.gamma(shape1, scale=scale1))
    mn2 = 10
    vr2 = 10
    scale2 = vr2 / mn2
    shape2 = mn2**2 / vr2
    s4 = uv(rv=ss.gamma(shape2, scale=scale2))
    s5 = uv(rv=ss.expon(scale=0.2))
    s6 = uv(rv=ss.chi2(10))
    T = s1 + s2 + s3 + s4 + s5 + s6
    print("Results should be about:")
    print(" > Mean...................  51.7")
    print(" > Variance...............  33.3")
    print(" > Skewness Coefficient...  0.52")
    print(" > Kurtosis Coefficient...  3.49")
    print(T)

    print("*" * 80)
    print("Example of two-bar truss")
    H = uv(rv=ss.norm(loc=30, scale=5 / 3.0), tag="H")
    B = uv(rv=ss.norm(loc=60, scale=0.5 / 3.0), tag="B")
    d = uv(rv=ss.norm(loc=3, scale=0.1 / 3), tag="d")
    t = uv(rv=ss.norm(loc=0.15, scale=0.01 / 3), tag="t")
    E = uv(rv=ss.norm(loc=30000, scale=1500 / 3.0), tag="E")
    rho = uv(rv=ss.norm(loc=0.3, scale=0.01 / 3.0), tag="rho")
    P = uv(rv=ss.norm(loc=66, scale=3 / 3.0), tag="P")
    pi = math.pi
    wght = 2 * pi * rho * d * t * umath.sqrt((B / 2) ** 2 + H**2)
    strs = (P * umath.sqrt((B / 2) ** 2 + H**2)) / (2 * pi * d * t * H)
    buck = (pi**2 * E * (d**2 + t**2)) / (8 * ((B / 2) ** 2 + H**2))
    defl = (P * ((B / 2) ** 2 + H**2) ** (1.5)) / (2 * pi * d * t * H**2 * E)
    print("wght:", wght)
    print("strs:", strs)
    print("buck:", buck)
    print("defl:", defl)
    wght.error_components(pprint=True)

    ###############################################################################

    print("*" * 80)
    print("SAME TEST FUNCTIONS USING EASY DISTRIBUTION CONSTRUCTORS")

    print("*" * 80)
    print("Example of a three part assembly")
    x1 = N(24, 1)  # normally distributed
    x2 = N(37, 4)  # normally distributed
    x3 = Exp(2)  # exponentially distributed
    Z = (x1 * x2**2) / (15 * (1.5 + x3))
    print("Results should be about:")
    print(" > Mean...................  1176.45")
    print(" > Variance...............  99699.682")
    print(" > Skewness Coefficient...  0.70801305")
    print(" > Kurtosis Coefficient...  6.1632855")
    print(Z)

    print("*" * 80)
    print("Example of volumetric gas flow through orifice meter")
    H = N(64, 0.5)
    M = N(16, 0.1)
    P = N(361, 2)
    t = N(165, 0.5)
    C = 38.4
    Q = C * umath.sqrt((520 * H * P) / (M * (t + 460)))
    print("Results should be about:")
    print(" > Mean...................  1330.9997")
    print(" > Variance...............  58.210763")
    print(" > Skewness Coefficient...  0.010942207")
    print(" > Kurtosis Coefficient...  3.0003269")
    print(Q)

    print("*" * 80)
    print("Example of manufacturing tolerance stackup")
    # for a gamma distribution we need the following conversions:
    # scale = var/mean
    # shape = mean**2/var
    mn = 1.5
    vr = 0.25
    scale = vr / mn
    shape = mn**2 / vr
    x = Gamma(shape, scale)
    y = Gamma(shape, scale)
    z = Gamma(shape, scale)
    w = x + y + z
    print("Results should be about:")
    print(" > Mean...................  4.5")
    print(" > Variance...............  0.75")
    print(" > Skewness Coefficient...  0.385")
    print(" > Kurtosis Coefficient...  3.22")
    print(w)

    print("*" * 80)
    print("Example of scheduling facilities (six stations)")
    s1 = N(10, 1)
    s2 = N(20, 2**0.5)
    mn1 = 1.5
    vr1 = 0.25
    scale1 = vr1 / mn1
    shape1 = mn1**2 / vr1
    s3 = Gamma(shape1, scale1)
    mn2 = 10
    vr2 = 10
    scale2 = vr2 / mn2
    shape2 = mn2**2 / vr2
    s4 = Gamma(shape2, scale2)
    s5 = Exp(5)
    s6 = Chi2(10)
    T = s1 + s2 + s3 + s4 + s5 + s6
    print("Results should be about:")
    print(" > Mean...................  51.7")
    print(" > Variance...............  33.3")
    print(" > Skewness Coefficient...  0.52")
    print(" > Kurtosis Coefficient...  3.49")
    print(T)

    print("*" * 80)
    print("Example of two-bar truss")
    H = N(30, 5 / 3.0, tag="H")
    B = N(60, 0.5 / 3.0, tag="B")
    d = N(3, 0.1 / 3, tag="d")
    t = N(0.15, 0.01 / 3, tag="t")
    E = N(30000, 1500 / 3.0, tag="E")
    rho = N(0.3, 0.01 / 3.0, tag="rho")
    P = N(66, 3 / 3.0, tag="P")
    pi = math.pi
    wght = 2 * pi * rho * d * t * umath.sqrt((B / 2) ** 2 + H**2)
    strs = (P * umath.sqrt((B / 2) ** 2 + H**2)) / (2 * pi * d * t * H)
    buck = (pi**2 * E * (d**2 + t**2)) / (8 * ((B / 2) ** 2 + H**2))
    defl = (P * ((B / 2) ** 2 + H**2) ** (1.5)) / (2 * pi * d * t * H**2 * E)
    print("wght:", wght)
    print("strs:", strs)
    print("buck:", buck)
    print("defl:", defl)
    wght.error_components(pprint=True)
