import numpy as np
import aifeynman

# generate power law:
x = np.linspace(0.1, 1.0, 1000)
y = 5*x**-0.5

# save to file:
with open('temp.txt', "w") as f:
    np.savetxt(f, np.array([x, y]).T)

aifeynman.run_aifeynman("./", "temp.txt", 60, "19ops.txt", polyfit_deg=2, NN_epochs=500)
