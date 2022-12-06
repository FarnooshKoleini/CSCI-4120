# HW6


## Group 28

Sepehr Saljooghi saljooghis19@students.ecu.edu<br>
Farnoosh Koleini Koleinif20@students.ecu.edu<br>
Richard Herget hergetr19@students.ecu.edu<br>

## Parameters selected for linear, radial, polynomial kernels.

Linear params:

- 'C': uniform(0,10)

Radial params:

- 'C': uniform(0,10)'
- 'gamma': uniform(0,0.01)

Polynomial params:

- 'C': uniform(0,1)
- 'gamma': uniform(0,0.01)
- 'degree': np.linspace(1,10,10)
- 'coef0': uniform(0,10)

## Results comparison between linear, radial, polynomial kernels

1) RBF (97%) 
2) Polynomial (96%) 
3) Linear (92%)
