# Report
1)DFT
a) Fast Fourier Transform:

A fast Fourier transform (FFT) algorithm computes the discrete Fourier transform (DFT) of a sequence, or its inverse (IFFT). 
Fourier analysis converts a signal from its original domain (often time or space) to a representation in the frequency domain and vice versa.

Using shape function, no of rows and columns of an input matrix can be captured. Then for every matrix value, we calculate the fast fourier transform using
the fft formula.
        
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
        fft_matrix=new_mat

fft_matrix has the final fft output.

b) Inverse Fourier Transform:
To calculate the inverse fourier transform, we take fft output as an input to this function, and the following calculations are performed

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
        ift_matrix=new_mat

ift_matrix has the final output for inverse fourier transform.
The outout of the inverse fourier transform values are not exactly same as the values calculated by inbuilt ifft2 function. But the relative 
difference between the values are same when compared.

c) Discrete Cosine Transform:
From the calculation of fourier transform, calculating only real part woould give result for discrete cosine transform.

d) Magnitude of Fourier transform:
Calculating the square root(pow(real,2)+pow(img,2)) gives us the maginitude of fourier transform which is given as an input matrix.
        
2) Filtering:

Filtering is a technique for modifying or enhancing an image. For example, you can filter an image to emphasize certain features or remove other features. 
Image processing operations implemented with filtering include smoothing, sharpening, and edge enhancement.

The following steps depicts the process involved in filtering:
		1. Compute the fft of the image
		   fft_image = np.fft.fft2(self.image)
		
        2. shift the fft to center the low frequencies
		   fft_shift = np.fft.fftshift(fft_image)
		   
		The result obtained here is DFT image. To view this image:
			a. Calculate the magnitude
			   dft_img=np.absolute(fft_shift)
			b. Do a logarithmic compression
		       dft_img=np.log(dft_img) * 10 (As I was getting very low values, i multiplied with 10 to show a significant image)
			c. Convert to uint8 to save it as greyscale image
		       dft_img=np.uint8(dft_img)
		
        3. get the mask calculated based on the given filter input
		   mask = self.filter(shape, cutoff) or mask = self.filter(shape,cutoff,order) for butterworth filter
		   
		   D(u,v) is the distance between a point (u,v) in the frequency domain and the center of the frequency rectangle (image)
		   
		   Algorithm followed for ideal low pass filter:
			if D(i,j) > cutoff:
				mask[i,j] = 0
			else:
				mask[i,j] = 1
		  
		   Algorithm followed for ideal high pass filter:
			mask = 1 - ideal_low_mask
		   
		   Formulae used for guassian low pass filter:
			mask[i,j]= exp( -pow(D(i,j),2)/2 * pow(cutoff,2) )
			
		   Formulae used for guassian high pass filter:
		    mask=1-guassian _low_mask
			
		   Formulae used for butterworth low pass filter:
			mask[i,j]=1/( 1 + pow(D(i,j)/cutoff,2*order))
		
		   Formulae used for butterworth high pass filter:
			mask = 1- butterworth_low_mask
			
		
		   
        4. filter the image frequency based on the mask (Convolution theorem)
		   filtered=fft_shift * mask
		   
		The result obtained at this step needs to be saved. Same process stated above is followed
				filtered_image=np.absolute(filtered) 		#magnitude
				filtered_image=np.log(filtered_image)*10  	#significant values
				
        5. compute the inverse shift
		   inv_shift = np.fft.ifftshift(filtered)
        6. compute the inverse fourier transform
		   inverse = np.fft.ifft2(inv_shift)
        7. compute the magnitude
		   magnitude = np.absolute(inverse)
		8. Perform Full contrast stretch
		   b=image.max()
           a=image.min()
		   new_mat=(255)*((image-a)/(b - a))
		The result obtained at this step is final filtered image.