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
        df=pd.read_csv(data_path)
        coords = df[['Line#', 'Column#']].values   # shape [num_pixels, 2]
        spectra = df.drop(columns=['Line#', 'Column#']).values  # shape [num_pixels, num_lambda]

        # Get spatial dimensions
        y_max = coords[:,0].max() + 1
        x_max = coords[:,1].max() + 1
        num_lambda = spectra.shape[1]

        # Initialize empty tensor
        tensor = torch.zeros((num_lambda, y_max, x_max), dtype=torch.float32)

        # Fill tensor
        for (y, x), spec in zip(coords, spectra):
            tensor[:, y, x] = torch.tensor(spec, dtype=torch.float32)

        return tensor
    
    def ref_normalization(self,cube,w_ref_path=None,d_ref_path=None,mode=1):
        """
        Method which will normalize the reflectance value with respect to slected mode.

        Input:
            w_ref_path: (string) path to cube contaning the white reference value
            d_ref_path: (string) path to cube contaning the black reference value
            cube: (torch.tensor) hypercube in torch tensor format 
            mode: (0 or 1) determine mode of normalization mode 0 will use additional data paths and mode
            
        """
        if mode==0:
            return cube
        elif mode==1:
            cube_norm=(cube-cube.min())/(cube.max()-cube.min())
            return cube_norm
        else:
            print('Not correct mode selected for the method!')
            return 1


        





        