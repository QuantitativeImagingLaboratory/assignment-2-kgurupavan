# For this part of the assignment, You can use inbuilt functions to compute the fourier transform
# You are welcome to use fft that are available in numpy and opencv
import numpy as np
import math as math
import cv2
class Filtering:
    image = None
    filter = None
    cutoff = None
    order = None

    def __init__(self, image, filter_name, cutoff, order = 0):
        """initializes the variables frequency filtering on an input image
        takes as input:
        image: the input image
        filter_name: the name of the mask to use
        cutoff: the cutoff frequency of the filter
        order: the order of the filter (only for butterworth
        returns"""
        self.image = image
        if filter_name == 'ideal_l':
            self.filter = self.get_ideal_low_pass_filter
        elif filter_name == 'ideal_h':
            self.filter = self.get_ideal_high_pass_filter
        elif filter_name == 'butterworth_l':
            self.filter = self.get_butterworth_low_pass_filter
        elif filter_name == 'butterworth_h':
            self.filter = self.get_butterworth_high_pass_filter
        elif filter_name == 'gaussian_l':
            self.filter = self.get_gaussian_low_pass_filter
        elif filter_name == 'gaussian_h':
            self.filter = self.get_gaussian_high_pass_filter

        self.cutoff = cutoff
        self.order = order



    def get_ideal_low_pass_filter(self, shape, cutoff):
        """Computes a Ideal low pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the ideal filter
        returns a ideal low pass mask"""
        row=self.image.shape[0]
        col=self.image.shape[1]
        mask=np.zeros((row,col), dtype=np.uint8)
        for i in range (0,row):
            for j in range (0,col):
                val = math.sqrt(math.pow((i-(row/2)),2) + math.pow((j-(col/2)),2))
                if val > cutoff:
                    mask[i,j]=0
                else:
                    mask[i,j]=1

        return mask


    def get_ideal_high_pass_filter(self, shape, cutoff):
        """Computes a Ideal high pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the ideal filter
        returns a ideal high pass mask"""

        #Hint: May be one can use the low pass filter function to get a high pass mask

        val=self.get_ideal_low_pass_filter(shape,cutoff)
        mask=1-val
        return mask

    def get_butterworth_low_pass_filter(self, shape, cutoff, order):
        """Computes a butterworth low pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the butterworth filter
        order: the order of the butterworth filter
        returns a butterworth low pass mask"""
        row = self.image.shape[0]
        col = self.image.shape[1]
        mask = np.zeros((row, col))
        power = 2 * order
        for i in range(0, row):
            for j in range(0, col):
                val = math.sqrt(math.pow((i - (row / 2)), 2) + math.pow((j - (col / 2)), 2))
                mask[i, j] = (1 / (1 + math.pow((val / cutoff), power)))

        return mask



    def get_butterworth_high_pass_filter(self, shape, cutoff, order):
        """Computes a butterworth high pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the butterworth filter
        order: the order of the butterworth filter
        returns a butterworth high pass mask"""

        #Hint: May be one can use the low pass filter function to get a high pass mask
        value = self.get_butterworth_low_pass_filter(shape, cutoff, order)
        mask = 1 - value

        return mask




    def get_gaussian_low_pass_filter(self, shape, cutoff):
        """Computes a gaussian low pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the gaussian filter (sigma)
        returns a gaussian low pass mask"""
        row = self.image.shape[0]
        col = self.image.shape[1]
        mask = np.zeros((row, col))
        for i in range(0, row):
            for j in range(0, col):
                val = math.sqrt(math.pow((i - (row / 2)), 2) + math.pow((j - (col / 2)), 2))
                mask[i, j] = math.exp(-math.pow(val,2)/(2*math.pow(cutoff,2)))

        return mask

    def get_gaussian_high_pass_filter(self, shape, cutoff):
        """Computes a gaussian high pass mask
        takes as input:
        shape: the shape of the mask to be generated
        cutoff: the cutoff frequency of the gaussian filter (sigma)
        returns a gaussian high pass mask"""

        #Hint: May be one can use the low pass filter function to get a high pass mask

        val=self.get_gaussian_low_pass_filter(shape,cutoff)
        mask=1-val
        return mask

    def post_process_image(self, image):
        """Post process the image to create a full contrast stretch of the image
        takes as input:
        image: the image obtained from the inverse fourier transform
        return an image with full contrast stretch
        -----------------------------------------------------
        1. Full contrast stretch (fsimage)
        2. take negative (255 - fsimage)
        """
        row=image.shape[0]
        col=image.shape[1]
       # new_mat=np.zeros((row,col))
        b=image.max()
        a=image.min()
        new_mat=(255)*((image-a)/(b - a))
       # for i in range(0,row):
        #    for j in range(0,col):
         #       value = round(255*((image[i,j]-1)/(b - a)) + 0.5)
                #value=(255*(image[i,j]-a))/(b-a)
          #      new_mat[i,j]=value
        image=new_mat
        return image


    def filtering(self):

        """Performs frequency filtering on an input image
        returns a filtered image, magnitude of DFT, magnitude of filtered DFT        
        ----------------------------------------------------------
        You are allowed to used inbuilt functions to compute fft
        There are packages available in numpy as well as in opencv
        Steps:
        1. Compute the fft of the image
        2. shift the fft to center the low frequencies
        3. get the mask (write your code in functions provided above) the functions can be called by self.filter(shape, cutoff, order)
        4. filter the image frequency based on the mask (Convolution theorem)
        5. compute the inverse shift
        6. compute the inverse fourier transform
        7. compute the magnitude
        8. You will need to do a full contrast stretch on the magnitude and depending on the algorithm you may also need to
        take negative of the image to be able to view it (use post_process_image to write this code)
        Note: You do not have to do zero padding as discussed in class, the inbuilt functions takes care of that
        filtered image, magnitude of DFT, magnitude of filtered DFT: Make sure all images being returned have grey scale full contrast stretch and dtype=uint8 
        """
        fft_image = np.fft.fft2(self.image)
        fft_shift = np.fft.fftshift(fft_image)
        dft_img=np.absolute(fft_shift)
        dft_img=np.log(dft_img) * 10
        dft_img=np.uint8(dft_img)

        shape=[fft_image.shape[0],fft_image.shape[1]]
        if(self.filter==self.get_ideal_low_pass_filter):
            mask = self.get_ideal_low_pass_filter(shape,self.cutoff)
        elif self.filter==self.get_ideal_high_pass_filter:
            mask = self.get_ideal_high_pass_filter(shape,self.cutoff)
        elif self.filter==self.get_butterworth_low_pass_filter:
            mask = self.get_butterworth_low_pass_filter(shape,self.cutoff,self.order)
        elif self.filter==self.get_butterworth_high_pass_filter:
            mask = self.get_butterworth_high_pass_filter(shape,self.cutoff,self.order)
        elif self.filter==self.get_gaussian_low_pass_filter:
            mask=self.get_gaussian_low_pass_filter(shape,self.cutoff)
        elif self.filter==self.get_gaussian_high_pass_filter:
            mask=self.get_gaussian_high_pass_filter(shape,self.cutoff)
        filtered=fft_shift * mask
        filtered_image=np.absolute(filtered)
        filtered_image=np.log(filtered_image)*10
        filtered_image=np.uint8(filtered_image)

        inv_shift = np.fft.ifftshift(filtered)

        inverse = np.fft.ifft2(inv_shift)

        magnitude = np.absolute(inverse)
        img = self.post_process_image(magnitude)

        #img = np.uint8(img)



        return [dft_img, filtered_image, img]
