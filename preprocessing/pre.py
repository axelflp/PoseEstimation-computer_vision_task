import numpy as np


# This function adjust the landmarks for a cut image following  the bbox annotation
def adjust_landmarks(data_list):
    for i in range(len(data_list)):

        bbox = data_list[i]['bbox']
        bb_x1 = int(bbox[0])
        bb_y1 = int(bbox[1])
        
        data_list[i].pop('visibility',None)
        
        x, y = data_list[i]['landmarks'][0::2], data_list[i]['landmarks'][1::2]
        
        x = np.array(x) - bb_x1
        y = np.array(y) - bb_y1
        
        land = np.array([x,y]).T.flatten()
        
        data_list[i].update({'landmarks':land})

    return