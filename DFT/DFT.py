# For this part of the assignment, please implement your own code for all computations,
# Do not use inbuilt functions like fft from either numpy, opencv or other libraries

import numpy as np
import scipy.fftpack
import math
class DFT:

    def forward_transform(self, matrix):
        """Computes the forward Fourier transform of the input matrix
        takes as input:
        matrix: a 2d matrix
        returns a complex matrix representing fourier transform"""
        #check_out=np.fft.fft2(matrix)
        #print(check_out)
        row = matrix.shape[0]
        col = matrix.shape[1]
        new_mat=np.zeros((row,col),dtype=np.complex)
        for u in range(0,row):
            for v in range(0,col):
                sum=0;
                for i in range(0,row):
                    for j in range(0,col):
                        real = math.cos(((2 * math.pi) / 15) * ((u * i) + (v * j)))
                        img = -math.sin(((2 * math.pi) / 15) * ((u * i) + (v * j)))
                        value= matrix[i,j] * complex(real,img)
                        sum= sum+value
                new_mat[u, v] = sum
        matrix=new_mat



        return matrix

    def inverse_transform(self, matrix):
        """Computes the inverse Fourier transform of the input matrix
        matrix: a 2d matrix (DFT) usually complex
        takes as input:
        returns a complex matrix representing the inverse fourier transform"""
        inverse=scipy.fftpack.ifft2(matrix)
        print(inverse)
        new_mat=np.zeros((15,15),dtype=np.complex)
        for i in range(0,15):
            for j in range(0,15):
                sum=0
                for u in range(0,15):
                    for v in range(0,15):
                        real = math.cos(((2 * math.pi) / 15) * ((u * i) + (v * j)))
                        img  = math.sin(((2 * math.pi) / 15) * ((u * i) + (v * j)))
                        value=matrix[u,v] * complex(real,img)
                        sum = sum + value
                new_mat[i,j] = sum
        matrix=new_mat
        return matrix


    def discrete_cosine_tranform(self, matrix):
        """Computes the discrete cosine transform of the input matrix
        takes as input:
        matrix: a 2d matrix
        returns a matrix representing discrete cosine transform"""
        row = matrix.shape[0]
        col = matrix.shape[1]
        new_mat = np.zeros((row, col))
        for u in range(0, row):
            for v in range(0, col):
                sum = 0;
                for i in range(0, row):
                    for j in range(0, col):
                        real = math.cos(((2 * math.pi) / 15) * ((u * i) + (v * j)))
                        value = matrix[i, j] * real
                        sum = sum + value
                new_mat[u, v] = sum
        matrix = new_mat


        return matrix


    def magnitude(self, matrix):
        """Computes the magnitude of the DFT
        takes as input:
        matrix: a 2d matrix
        returns a matrix representing magnitude of the dft"""
        row = matrix.shape[0]
        col = matrix.shape[1]
        new_mat = np.zeros((row, col))
        for i in range(0, row):
            for j in range(0, col):
                new_mat[i,j]=np.absolute(matrix[i,j])
        return new_mat