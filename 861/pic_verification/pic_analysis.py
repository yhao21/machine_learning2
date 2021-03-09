import pandas as pd
import imageio, glob, os, re
import numpy as np
from sklearn.mixture import GaussianMixture as gmm
import matplotlib.pyplot as plt



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
    # go through each pictures
    # one_file stands for the file path
    df_list = []
    for one_file in glob.glob(os.path.join(os.getcwd(), 'data', '*.jpeg')):
        image_one = get_rgb(one_file).tolist()
        file_name = re.compile(r'data/(.*?)\.jpeg').findall(one_file)[0]
        ## Notice, image_one is a list, so you must append file name to image_one first
        image_one.append(file_name)
        df_list.append(image_one)

        

    df = pd.DataFrame(df_list, columns = ['R', 'G', 'B', 'name'])
    print(df)


    x1 = df.iloc[:, 0].values
    x2 = df.iloc[:, 1].values
    x3 = df.iloc[:, 2].values
    plt.plot(x1, x2, '.')
    plt.savefig('color_x1_x2.png')
    plt.clf()

    plt.plot(x1, x3, '.')
    plt.savefig('color_x1_x3.png')
    plt.clf()


    plt.plot(x2, x3, '.')
    plt.savefig('color_x2_x3.png')
    plt.clf()










    
    
