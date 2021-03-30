import pandas as pd
import imageio, glob, os, re
import numpy as np
from sklearn.mixture import GaussianMixture as gmm
import matplotlib.pyplot as plt
from sklearn import preprocessing
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering



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
    ## notice, glob may not go through with order, i.e., pic1, pic2,.. it will be pic3, pic5, pic1, ...
    for one_file in glob.glob(os.path.join(os.getcwd(), 'data', '*.jpeg')):
        image_one = get_rgb(one_file).tolist()
        file_name = re.compile(r'data/(.*?)\.jpeg').findall(one_file)[0]
        ## Notice, image_one is a list, so you must append file name to image_one first
        image_one.append(file_name)
        df_list.append(image_one)

        

    df = pd.DataFrame(df_list, columns = ['R', 'G', 'B', 'name'])
    #print(df)


    x1 = df.iloc[:, 0].values
    x2 = df.iloc[:, 1].values
    x3 = df.iloc[:, 2].values

    ### plotting
    #plt.plot(x1, x2, '.')
    #plt.savefig('color_x1_x2.png')
    #plt.clf()

    #plt.plot(x1, x3, '.')
    #plt.savefig('color_x1_x3.png')
    #plt.clf()


    #plt.plot(x2, x3, '.')
    #plt.savefig('color_x2_x3.png')
    #plt.clf()

    ##sort the value
    #df = df.sort_values(by=['name'])
    #print(df)
    

    '''
    ahc assume all files in different gourps at beginning
    it group two in one group only if the distance between them are the most
    minimum.
    The program keep doing this untile there's only one group
    '''
    df = df.sort_values(['name'])
    print(df)

    dataset = pd.DataFrame(preprocessing.normalize(df.iloc[:, 0:3]))
    dataset.columns = ['r', 'g', 'b']
    machine = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
    results = machine.fit_predict(dataset)
    df['ahc_result'] = results

    #plotting ahc result
    plt.scatter(dataset['r'], dataset['g'], c = results)
    plt.savefig('ahc_clustering.png')
    plt.clf()


    plt.title('Dendrogram')
    dendrogram_obj = shc.dendrogram(shc.linkage(dataset, method = 'ward'))
    plt.savefig('den.png')
    plt.close()
    '''
    horizontal line in dendrogram means the row number. 
    i.e., row 1 and row 43 are in same group they are pic02 and pic44

    Vertical axis stands for the distance
    From dendrogram, ahc ways we should only have 3 groups.
    If grouping of two pictures does not increase distance alot, then we should
    group them together.

    so, we should avoid those blue line, hence we only have orange, red and green
    three groups
    '''


    

    
    
