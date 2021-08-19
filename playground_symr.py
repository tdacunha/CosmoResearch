import numpy as np
import tempfile
import os

# generate power law:
x = np.linspace(0.1, 1.0, 1000)
y = 5*x**-0.6

###############################################################################
# GPlearn

from gplearn.genetic import SymbolicRegressor
from gplearn import functions

def exponential(x):
    return np.exp(x)

def _protected_exponent(x1):
    with np.errstate(over='ignore'):
        return np.where(np.abs(x1) < 100, np.exp(x1), 0.)

def _pow(x1,x2):
    with np.errstate(over='ignore', invalid='ignore'):
        return np.where(np.logical_and(x2 < 10, x2 < 1000), np.power(x1, x2), 1.)


myexp = functions.make_function(function=_protected_exponent,
                        name='exp',
                        arity=1)

mypow = functions.make_function(_pow,
                        name='pow',
                        arity=2)

est_gp = SymbolicRegressor(function_set=['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', myexp],
                           population_size=5000,
                           generations=20, stopping_criteria=0.01,
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           max_samples=0.9, verbose=1,
                           parsimony_coefficient=0.01, random_state=0)
est_gp.fit(x.reshape(1, -1).T, y)

print(est_gp._program)
import sympy
from sympy import sympify, simplify
locals = {
    'sub': lambda x, y : x - y,
    'div': lambda x, y : x/y,
    'mul': lambda x, y : x*y,
    'add': lambda x, y : x + y,
    'neg': lambda x    : -x,
    'pow': lambda x, y : x**y,
    'cos': lambda x    : sympy.cos(x),
    'log': lambda x    : sympy.log(x),
}
simplify(sympify(str(est_gp._program), locals=locals))


###############################################################################
# AI-feynman

"""
import aifeynman
# create temp folder:
with tempfile.TemporaryDirectory() as dirpath:
    # go to the temp directory:
    os.chdir(dirpath)
    # save to file:
    with open(dirpath+'/temp.txt', "w") as f:
        np.savetxt(f, np.array([x, y]).T)
    # run AI-Feynman:
    aifeynman.run_aifeynman(dirpath+'/', 'temp.txt', 60, 'marco_ops.txt', polyfit_deg=2, NN_epochs=500)
"""
