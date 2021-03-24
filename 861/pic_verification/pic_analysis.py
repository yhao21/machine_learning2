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
    '''
    imimage_process = imimage/255

    if you print imimage_process now, it gives you

[[[0.08627451 0.29803922 0.72156863]    
  [0.08627451 0.29803922 0.72156863]    
  [0.08627451 0.29803922 0.72156863]    
  ...                                     
  [0.38823529 0.70588235 0.96078431]    
  [0.38431373 0.70196078 0.95686275]    
  [0.38039216 0.69803922 0.95294118]]   


 [[0.08627451 0.29803922 0.72156863]    
  [0.08627451 0.29803922 0.72156863]    
  [0.08627451 0.29803922 0.72156863]    
  ...                                     
  [0.41176471 0.7254902  0.97254902]    
  [0.40784314 0.72156863 0.96862745]    
  [0.40392157 0.71764706 0.96470588]]   

 ...

 [[0.42745098 0.6627451  0.89803922]
  [0.43137255 0.66666667 0.90196078]
  [0.43529412 0.67058824 0.90588235]
  ...
  [0.80392157 0.90588235 0.99607843]
  [0.80392157 0.90588235 0.99607843]
  [0.80392157 0.90588235 0.99607843]]]


    if you print the shape for each pic's matrix, 

    (407, 612, 3)   # pic1
    (408, 612, 3)   # pic2
    (408, 612, 3)   # pic3

    if you print imimage_process[0] you will get the first block:
    [[[0.08627451 0.29803922 0.72156863]    
      [0.08627451 0.29803922 0.72156863]    
      [0.08627451 0.29803922 0.72156863]    
      ...                                     
      [0.38823529 0.70588235 0.96078431]    
      [0.38431373 0.70196078 0.95686275]    
      [0.38039216 0.69803922 0.95294118]]   

    Then, imimage_process.sum(axis = 0) will add rows together to one row.
    It is like this, row 1 + row2 + row3 + ...
    [0.08627451 0.29803922 0.72156863] + [0.08627451 0.29803922 0.72156863] + ...
    
    So we end up with one row with three columns (R, G, B)
    This behavior only aggregate these 612 subrows in (407, 612, 3)   # pic1,
    Notice, this picture has 407 rows, and each rows has 612 subrows,
    so we have to add these rows again, that is why we have two .sum(axis = 0)

    It would be: imimage_process = imimage_process.sum(axis = 0).sum(axis = 0)
    Now, imimage_process for this picture ends up with one row and 3 columns.
    But the value of each column is a aggregate num rather a fraction.
    Since we have compress two dimension in to one, to get a fraction, we need
    to divid it by the product of len(horizontal) and len(vertical)

    It would be more reasonable if we take a look at this example:

    a = [[[1   2   3]
          [1   1   1]
          [1   2   2]]
         [[1   2   3]
          [1   1   1]
          [1   2   2]]
     if we do a.sum(axis = 0), we would have two rows and three columns because
    [1   2   3]     [1   2   3]
    [1   1   1]  +  [1   1   1]
    [1   2   2]     [1   2   2]

    numpy do this:
    [1   2   3]  +  [1   2   3] = [2   4   6]
    [1   1   1]  +  [1   1   1] = [2   2   2]
    [1   2   2]  +  [1   2   2] = [2   4   4]
    
    so we end up with 
    a = [[2 4 6]
         [2 2 2]
         [2 4 4]]

    Then we do sum again, we would have 
    a = [6, 10, 12]

    to make each element a fraction, we need to divid each value by there
    orginial dimension, then we end up with three percentage number.
    notice, a.shape = (2, 3, 3)
    2 : we have two sub matrix,
    3 : in each matrix, we have three rows
    3 : in each row, we have three element. (or the matrix has three columns)

    so we divid each element in a = [6, 10, 12] by 2*3 = 6
    a = [6/6]



    imimage_process = imimage_process.sum(axis = 0).sum(axis = 0)/imimage_process.shape[0]*imimage_process.shape[1]


    if you
    '''







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

    gmm_data = df.iloc[:, 0:3].values
    ## use sklearn method to normalize
    gmm_data = preprocessing.normalize(gmm_data)
    machine = gmm(n_components=4)
    machine.fit(gmm_data)
    gmm_result = machine.predict(gmm_data)
    #print(gmm_result)

    plt.scatter(x1,x2,c = gmm_result)
    plt.savefig('gmm_result.png')
    df['gmm_result'] = gmm_result
    #print(df)


    #sort the value
    df = df.sort_values(by=['name'])
    print(df)






    
    
