import numpy as np 
import threading
import os
import glob 
from skimage.io import imread
from random import shuffle
import cv2


class Generator:  
    
    def __init__(self, directory, batch_size, yuv=False, normalize=True, SVD=False, flip=False):
        
        self.directory = directory
        self.batch_size = batch_size
        self.img_dir = None
        self.low_rank_imgs = None
        self.lock = threading.Lock()
        
        #setup data
        self.setup_data()          
        self.setup_lowranks()
        
        #for frame shift of images in list img_dir
        self.start = 0
        self.end = self.batch_size 
        
        #init output batch
        self.X_batch = np.zeros((batch_size, 180, 320, 3))
        self.y_batch = np.zeros((batch_size, 1))
        
        #augment 
        self.SVD = SVD
        self.flip = flip 
        self.normalize = normalize
        self.yuv = yuv

    def setup_data(self):
        '''get and verify images are in directory, labels are correct '''
        
        self.img_dir = glob.glob(self.directory + '/images/*')
        shuffle(self.img_dir)
        self.num_batches = int(np.ceil(len(self.img_dir)/ self.batch_size)) #number of batches 

        assert len(self.img_dir) > 0, 'no images found, check directory'

    def setup_lowranks(self):
        '''setup the preprocessed SVD lowrank images
        build a dict of names : image tensors'''
        
        self.low_rank_imgs = {}
        img_list = glob.glob(self.directory + '/*low_rank.png')
        
        for im in img_list: 
            #same name formatting 
            name = im.replace(self.directory, '')
            name = name.replace('\\', '')
            name = name.split('_')[:2]
            name = '{}_{}'.format(name[0], name[1])  
            self.low_rank_imgs[name] = imread(im)
    
    def get_img_metadata(self, img_name):
        '''get metadata from image name 
        returns a dict with keys: 
        
        name : (str) video name
        frame : (int) frame number 
        label : (float) steering angle
        array : image tensor'''
        
        unique_name = img_name.split('epoch')[-1].split('_')
        label = unique_name[-1].split('.')[-2]
        vid_name = 'epoch' + unique_name[0] + '_' + unique_name[1]
        
        
        d = {}
        d['label'] = float(label)/10
        d['frame'] = int(unique_name[-2])
        d['name'] = vid_name
        d['array'] = imread(img_name) #self.rgb2gray()
        
        return d
    
    def remove_background_SVD(self, img_data):
        '''get low rank image from frame's parent video 
        subtract from image array, in place for dict. 
        Apply weighed average of frame and low-rank matrix
        '''
        low_rank = self.low_rank_imgs[img_data['name']]
        diff = 255- cv2.absdiff(img_data['array'] , low_rank)
        img_data['array']  = cv2.addWeighted(diff, 0.9, img_data['array'], 0.1, 1)

        
    def flip_img(self, img_data):
        '''randomly flip the image and reverse the steering angle
        50/50 odds'''
        if np.random.randint(0, 2): 
            img_data['array'] = np.flip(img_data['array'], axis=1)
            img_data['label'] *= -1 
            
    def normalize_img(self, img_data): 
        '''normalize the image tensor [-1, 1]'''
        mini = np.min(img_data['array'])
        maxi = np.max(img_data['array'])
        img_data['array'] = img_data['array']/255
        #img_data['array'] = 2*((img_data['array'] - mini) / (maxi-mini))-1
        
    def cvt_YUV(self, img_data):
        '''cvt color space from RGB to YUV'''
        img_data['array'] = cv2.cvtColor(img_data['array'], cv2.COLOR_BGR2YUV)
    

    def __next__(self): 
        '''Yields data tensor of size [batch_size, 180, 320, 1], 
        label tensor of size [batch_size, 1]. GPU compatible. '''

        #lock and release threads at iteration execution 
        with self.lock:      
            for i in range(self.num_batches):
                img_batch_files = self.img_dir[self.start:self.end]

                for j, img_name in enumerate(img_batch_files): 
                    img_data = self.get_img_metadata(img_name) #image data dict
                     
                    #augment image
                    if self.SVD: 
                        self.remove_background_SVD(img_data)
                    if self.flip:
                        self.flip_img(img_data)
                    if self.yuv: 
                        self.cvt_YUV(img_data)
                    if self.normalize:
                        self.normalize_img(img_data)
                    
                        
                    self.X_batch[j, :, :, :] = img_data['array'].reshape(180,320,-1)
                    self.y_batch[j] = img_data['label'] #get steering angle

                #clip last batch 
                if i == self.num_batches - 1:
                    self.X_batch = self.X_batch[:j, :, :, :]       

                #increment images for next iteration 
                self.start += self.batch_size
                self.end += self.batch_size
                
                return  self.X_batch, self.y_batch
                
    def __iter__(self):
        return self