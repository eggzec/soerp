"""
Generalizes mathematical operators that work on numeric objects (from the math
module) compatible with objects with uncertainty distributions
"""
from soerp import _make_UF_compatible_object
from ad.admath import admath
#import sys

__author__ = 'Abraham Lee'

__all__ = []

e = admath.e
pi = admath.pi

__all__.append('e')
__all__.append('pi')

def abs(x):
    return _make_UF_compatible_object(admath.abs(x))
__all__.append('abs')

def acos(x):
    return _make_UF_compatible_object(admath.acos(x))
__all__.append('acos')

def acosh(x):
    return _make_UF_compatible_object(admath.acosh(x))
__all__.append('acosh')

def acot(x):
    return _make_UF_compatible_object(admath.acot(x))
__all__.append('acot')

def acoth(x):
    return _make_UF_compatible_object(admath.acoth(x))
__all__.append('acoth')

def acsc(x):
    return _make_UF_compatible_object(admath.acsc(x))
__all__.append('acsc')

def acsch(x):
    return _make_UF_compatible_object(admath.acsch(x))
__all__.append('acsch')

def asec(x):
    return _make_UF_compatible_object(admath.asec(x))
__all__.append('asec')

def asech(x):
    return _make_UF_compatible_object(admath.asech(x))
__all__.append('asech')

def asin(x):
    return _make_UF_compatible_object(admath.asin(x))
__all__.append('asin')

def asinh(x):
    return _make_UF_compatible_object(admath.asinh(x))
__all__.append('asinh')

def atan(x):
    return _make_UF_compatible_object(admath.atan(x))
__all__.append('atan')

def atan2(y, x):
    return _make_UF_compatible_object(admath.atan2(y, x))
__all__.append('atan2')

def atanh(x):
    return _make_UF_compatible_object(admath.atanh(x))
__all__.append('atanh')

def ceil(x):
    return _make_UF_compatible_object(admath.ceil(x))
__all__.append('ceil')

def cos(x):
    return _make_UF_compatible_object(admath.cos(x))
__all__.append('cos')

def cosh(x):
    return _make_UF_compatible_object(admath.cosh(x))
__all__.append('cosh')

def cot(x):
    return _make_UF_compatible_object(admath.cot(x))
__all__.append('cot')

def coth(x):
    return _make_UF_compatible_object(admath.coth(x))
__all__.append('coth')

def csc(x):
    return _make_UF_compatible_object(admath.csc(x))
__all__.append('csc')

def csch(x):
    return _make_UF_compatible_object(admath.csch(x))
__all__.append('csch')

def degrees(x):
    return _make_UF_compatible_object(admath.degrees(x))
__all__.append('degrees')

def erf(x):
    return _make_UF_compatible_object(admath.erf(x))
__all__.append('erf')

def erfc(x):
    return _make_UF_compatible_object(admath.erfc(x))
__all__.append('erfc')

def exp(x):
    return _make_UF_compatible_object(admath.exp(x))
__all__.append('exp')

def expm1(x):
    return _make_UF_compatible_object(admath.expm1(x))
__all__.append('expm1')

def fabs(x):
    return _make_UF_compatible_object(admath.fabs(x))
__all__.append('fabs')

def factorial(x):
    return _make_UF_compatible_object(admath.factorial(x))
__all__.append('factorial')

def floor(x):
    return _make_UF_compatible_object(admath.floor(x))
__all__.append('floor')

def gamma(x):
    return _make_UF_compatible_object(admath.gamma(x))
__all__.append('gamma')

def lgamma(x):
    return _make_UF_compatible_object(admath.lgamma(x))
__all__.append('lgamma')

def hypot(x,y):
    return _make_UF_compatible_object(admath.hypot(x,y))
__all__.append('hypot')

def ln(x):
    return _make_UF_compatible_object(admath.ln(x))
__all__.append('ln')

def log(x, base):
    return _make_UF_compatible_object(admath.log(x, base))
__all__.append('log')

def log10(x):
    return _make_UF_compatible_object(admath.log10(x))
__all__.append('log10')

def log1p(x):
    return _make_UF_compatible_object(admath.log1p(x))
__all__.append('log1p')

def pow(x):
    return _make_UF_compatible_object(admath.pow(x))
__all__.append('pow')

def radians(x):
    return _make_UF_compatible_object(admath.radians(x))
__all__.append('radians')

def sec(x):
    return _make_UF_compatible_object(admath.sec(x))
__all__.append('sec')

def sech(x):
    return _make_UF_compatible_object(admath.sech(x))
__all__.append('sech')

def sin(x):
    return _make_UF_compatible_object(admath.sin(x))
__all__.append('sin')

def sinh(x):
    return _make_UF_compatible_object(admath.sinh(x))
__all__.append('sinh')

def sqrt(x):
    return _make_UF_compatible_object(admath.sqrt(x))
__all__.append('sqrt')

def tan(x):
    return _make_UF_compatible_object(admath.tan(x))
__all__.append('tan')

def tanh(x):
    return _make_UF_compatible_object(admath.tanh(x))
__all__.append('tanh')

def trunc(x):
    return _make_UF_compatible_object(admath.trunc(x))
__all__.append('trunc')


    
