# makes numpy available
import numpy as np
# makes the eigen- calculating method available
from numpy.linalg import eig

# creates a small example array
array = np.array([[1, 7452, 0.002463],
                  [7452, 1, 4],
                  [0.002463, 4, 1]])

# uses the method to calculate the eigenvalues/eigenvectors
eigenvalue,eigenvector=eig(array)

# prints the eigenvalues (a 2x1 array)
print('Eigenvalues:', eigenvalue)
# prints the eigenvectors (a 2x2 array)
print('Eigenvectors:', eigenvector)

magnitudes=[]

# iterates through the first axis
for i in range(len(eigenvector)):
    # creates a temporary variable to hold the magnitudes
    magnitude_i = 0

    # iterates over the second axis
    for j in range(len(eigenvector[i])):
        # gets the coefficient for a given set of coordinates (i, j)
        coefficient_ij = eigenvector[i,j]
        # squares this and adds it to the temporary valye
        magnitude_i = magnitude_i+(coefficient_ij**2)

    # gets the square root and puts it into an array
    magnitudes.append(np.sqrt(magnitude_i))

# prints the magnitudes
print('List of the magnitudes of the eigenvectors:', magnitudes)

