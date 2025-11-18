import pandas as pd
import numpy as np
import torch

class Hypercube:
    """
    Class which handles data conversion and application of the hyperspectral data from Headwall VIS and SWIR data
    """
    def __init__(self,d_type=torch.float):
        self.d_type=d_type

    def reshape_3d(self,data_path):
        """
        Method which convert data format of headwall camera csv file into thetorch 3d tensor

        input:
            data_path: (string of data path

        output:
            cube: (torch.tensor) 3D hypercube with dimmension [s,y,x]
        """
        # Reading csv file with hyperspectral photos
        data=pd.read_csv(data_path)
        y_range=data['Line#'].max()+1
        x_range=data['Column#'].max()+1

        # Converting it into the numpy array
        data_np=data.values
        # Removing indexing of the coordinates
        data_np=data_np[:,2:]
        # Reshaping it into the 3D hypercube
        data_np=data_np.reshape(-1,y_range,x_range)

        # Converting it into the torch tensor
        cube=torch.from_numpy(data_np).to(self.d_type)

        return cube





        