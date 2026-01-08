import sys,os
import math

SIGMA_Q =   float(sys.argv[1])    
SIGMA_D =   float(sys.argv[2])    
D_VS    =   float(sys.argv[3])*10.0
DIR     =   sys.argv[4]

RMIN_QSI    = SIGMA_Q*(2**(1/6))
RMIN        = SIGMA_D*(2**(1/6))
D_Q         = (RMIN_QSI-RMIN)/(2*0.9724)*10.0

D_Q_a = D_Q / 3 * math.sqrt(6) 
D_VS_a = D_VS / 3 * math.sqrt(6)
factor = 1/math.sqrt(2)

with open('RSi.mol', 'w') as wf:
    wf.write("""
# Water molecule. Explicit TIP4P geometry for use with fix rigid

9 atoms
14 bonds

Coords

1 {:8.3f}{:8.3f}{:8.3f}
2 {:8.3f}{:8.3f}{:8.3f}
3 {:8.3f}{:8.3f}{:8.3f}
4 {:8.3f}{:8.3f}{:8.3f}
5 {:8.3f}{:8.3f}{:8.3f}
6 {:8.3f}{:8.3f}{:8.3f}
7 {:8.3f}{:8.3f}{:8.3f}
8 {:8.3f}{:8.3f}{:8.3f}

Types

1       1   # Si
2       2   # O
3       2   # O
4       2   # O
5       2   # O
6       3   # V
7       3   # V
8       3   # V
9       3   # V

Bonds

1       1  1  2
2       1  1  3
3       1  1  4
4       1  1  5
5       2  2  3
6       2  2  4
7       2  2  5
8       2  3  4
9       2  3  5
10      2  4  5
11      3  1  6
12      3  1  7
13      3  1  8
14      3  1  9

Special Bond Counts

1 8 0 0
2 4 4 0
3 4 4 0
4 4 4 0
5 4 4 0
6 1 7 0
7 1 7 0
8 1 7 0
9 1 7 0

Special Bonds

1 2 3 4 5 6 7 8 9
2 1 3 4 5 6 7 8 9
3 1 2 4 5 6 7 8 9
4 1 2 3 5 6 7 8 9
5 1 2 3 4 6 7 8 9
6 1 2 3 4 5 7 8 9
7 1 2 3 4 5 6 8 9
8 1 2 3 4 5 6 7 9
9 1 2 3 4 5 6 7 8
""".format( D_Q_a,  0.0, -D_Q_a*factor, \
           -D_Q_a,  0.0, -D_Q_a*factor, \
            0.0,  D_Q_a,  D_Q_a*factor, \
            0.0, -D_Q_a,  D_Q_a*factor, \
           -D_VS_a,  0.0,  D_VS_a*factor, \
            D_VS_a,  0.0,  D_VS_a*factor, \
            0.0, -D_VS_a, -D_VS_a*factor, \
            0.0,  D_VS_a, -D_VS_a*factor))

