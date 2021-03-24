import pandas as pd
import imageio, glob, os, re
import numpy as np
from sklearn.mixture import GaussianMixture as gmm
import matplotlib.pyplot as plt
from sklearn import preprocessing



'''
install package: imageio
pip3.8 install imageio
'''

def get_rgb(file_path):
    ### pilmode: each element contains 3 numbers representing Red, Green, Blue.
    ### each color from 0 to 255 (2^8) (it is 8 binary number, e.g. 10010101)
    imimage = imageio.imread(file_path, pilmode = 'RGB')
    #print(imimage)
    
    ### normalized the data
    ### since each from 0 to 255, to get a fraction from 0 to 1, 
    imimage_process = imimage/255
    print(imimage_process.shape)

    #print('\n\n')
    #print(imimage_process[0])
    #print('\n\n')
    #a = 0
    #for row in range(len(imimage_process)):
    #    a += imimage_process[row][0]
    #print(a)
    #print('\n\n')
    #print(imimage_process.sum(axis = 0))


    #print(imimage_process)
    #print(imimage_process.sum(axis = 0).shape)
    ### denominators: # of col times # of row = the size (total number)
    imimage_process = imimage_process.sum(axis = 0).sum(axis = 0)/imimage_process.shape[0]*imimage_process.shape[1]
    ##                              R               G               B
    ### before normalization: [136202.83529412 233951.60588235 337967.68235294]
    
    ### normalization vector
    imimage_process = imimage_process/np.linalg.norm(imimage_process, ord = None)
    ##                           R           G           B
    ### after normalization:[0.31454134 0.54027841 0.78048894]
    
    return imimage_process
    

if __name__ == '__main__':
#    # go through each pictures
#    # one_file stands for the file path
#    df_list = []
#    ## notice, glob may not go through with order, i.e., pic1, pic2,.. it will be pic3, pic5, pic1, ...
#
#    #file_path = os.path.join('data', 'pic01.jpeg')
#    #print(file_path)
#    #image_one = get_rgb(file_path).tolist()
#
#
#
#
#
#
#
#    for one_file in glob.glob(os.path.join(os.getcwd(), 'data', '*.jpeg')):
#        image_one = get_rgb(one_file).tolist()
#        file_name = re.compile(r'data/(.*?)\.jpeg').findall(one_file)[0]
#        ## Notice, image_one is a list, so you must append file name to image_one first
#        image_one.append(file_name)
#        df_list.append(image_one)
#
#        
#
#    df = pd.DataFrame(df_list, columns = ['R', 'G', 'B', 'name'])





    a = np.array([[[1, 2,3],
                   [1, 1,1],
                   [1, 2,2]],
                  [[1, 2,3],
                   [1, 1,1],
                   [1, 2,2]]])
    print(a.shape)
    print(a.shape[0] * a.shape[1])
    print(a.sum(axis = 0).sum(axis = 0)/a.shape[0]*a.shape[1])
