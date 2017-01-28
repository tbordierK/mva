import numpy as np

def convolution2dSepatarated(img,filtre_v,filtre_h):
	'''
	Computes the convolution of an image with a 2D
	separation filter
	The mirror rule was used for boundaries
	'''

	h,w = img.shape
	img_2 = np.copy(img)
	

	for i in range(h):
	    for j in range(w):
	        img_2[i,j] = sum(img[:,j]*np.roll(filtre_v,i-h/2))
	  
	img_3 = np.copy(img)      
	for i in range(h):
	    for j in range(w):
	        img_3[i,j] = sum(img_2[i,:]*np.roll(filtre_h,j-w/2))

	return img_3



def atrou_filter(filtre,j):
	'''
	Transforms filter to a trous version
	'''
	new_filtre = np.copy(filtre)

	for i in range(256):
		u = (1.*(i-256/2)/(2**j))
		if u.is_integer():
			new_filtre[i] = filtre[int(u)+256/2]
		else:
			new_filtre[i]= 0
        
	return new_filtre


def forWaveTF(img,filtre,J):
	'''
	the "a trous" algorithm to perform the
	forward isotropic wavelet transform
	'''
	c = np.array(np.array([img for k in range(J+1)]))
	w = np.array(np.array([img for k in range(J+1)]))
	for j in range(J):
	    atrou_filtre = atrou_filter(filtre,j)
	    new_img = convolution2dSepatarated(c[j],atrou_filtre,atrou_filtre)
	    c[j+1,:,:] = new_img
	    w[j+1,:,:] = c[j] - c[j+1]

	return c,w

def backWaveTF(c,w):
	'''
	backward isotropic wavelet transform.
	'''
	
	img = c[-1]+sum(w[1:])

	return img

