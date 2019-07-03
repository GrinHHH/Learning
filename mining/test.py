import numpy as np

def SampEn(U, m, r):

     def _maxdist(x_i, x_j):
           return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

     def _phi(m):
          x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
          B = [(len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) - 1.0) / (N - m) for x_i in x]
          return (N - m + 1.0)**(-1) * sum(B)

     N = len(U)

     return -np.log(_phi(m+1) / _phi(m))

# Usage example
U = np.array([1,2, 89,4]*2 )
print(SampEn(U,2,3))
