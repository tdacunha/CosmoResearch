import numpy as np
import aifeynman
import tempfile
import os

# generate power law:
x = np.linspace(0.1, 1.0, 1000)
y = 5*x**-0.6
z = np.ones(1000)

# create temp folder:
with tempfile.TemporaryDirectory() as dirpath:
    # go to the temp directory:
    os.chdir(dirpath)
    # save to file:
    with open(dirpath+'/temp.txt', "w") as f:
        np.savetxt(f, np.array([x, y]).T)
    # run AI-Feynman:
    aifeynman.run_aifeynman(dirpath+'/', 'temp.txt', 60, 'marco_ops.txt', polyfit_deg=2, NN_epochs=500)
