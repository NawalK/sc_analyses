import numpy as np
import pandas as pd
import os.path
import nibabel as nib

class Lesions:
    '''
    The Lesions class is used to store various measures regarding stroke lesion

    Attributes
    ----------
    config : dict
        Contains information regarding subjects, sessions, masks, etc.
    lesions: df
        Contains the informations on lesions related to the config file
            - weighted corticospinal load for both sides
            - total volume of the lesion
    '''

    def __init__(self, config):
        self.config = config

    def parse_lesions(self):
        '''Extract information on lesions and put them into a dataframe
            - weighted CST load (L/R)
            - total lesion volume 

        Returns
        ----------
        lesions : df
            Contains lesion infos
        '''   
        subs = []
        sessions = []
        weighted_load_L = []
        weighted_load_R = []
        lesion_vol = []

        # Loop through patients only
        patients_only = (sub for sub in self.config['list_subjects'] if self.config['list_subjects'][sub]['subtype'] == 'P')
        for sub in patients_only:  
            for session in self.config['list_subjects'][sub]['sess']:
                subs.append(sub)
                sessions.append(session)
                tmp_load = np.zeros((2, 1))
                for sideix, side in enumerate(['L', 'R']):
                    if self.config['acute_only'] == True and os.path.isfile(self.config['lesion_dir'] + 'Lesions/' + sub + '-' + session + '-acute_only-overlap_FSL_CST_' + side + '.nii.gz'):
                        lesion_overlap = nib.load(self.config['lesion_dir'] + 'Lesions/' + sub + '-' + session + '-acute_only-overlap_FSL_CST_' + side + '.nii.gz')
                    else:
                        lesion_overlap = nib.load(self.config['lesion_dir'] + 'Lesions/' + sub + '-' + session + '-overlap_FSL_CST_' + side + '.nii.gz')
                    lesion_overlap_np = lesion_overlap.get_fdata()
                    if side == 'L':
                        weighted_load_L.append(_compute_weighted_load(lesion_overlap_np, sideix, self.config['lesion_dir']))
                    elif side == 'R':
                        weighted_load_R.append(_compute_weighted_load(lesion_overlap_np, sideix, self.config['lesion_dir']))

                # Add also total volume of lesion
                if self.config['acute_only'] == True and os.path.isfile(self.config['lesion_dir'] + 'Lesions/' + sub + '-' + session + '-acute_only-overlap_FSL_CST_' + side + '.nii.gz'):
                    lesion = nib.load(self.config['lesion_dir'] + 'Lesions/' + sub + '-' + session + '-lesion_acute_only.nii')
                else:
                    lesion = nib.load(self.config['lesion_dir'] + 'Lesions/' + sub + '-' + session + '-lesion.nii')
                lesion_np = lesion.get_fdata()
                lesion_vol.append(np.sum(lesion_np))
                del tmp_load

        colnames = ["sub", "sess", "CST_L", "CST_R", "vol"]
        lesions_df = pd.DataFrame(list(zip(subs, sessions, weighted_load_L, weighted_load_R, lesion_vol)), columns=colnames)
        
        return lesions_df


# Function for CST load computation
def _compute_weighted_load(lesion, side, atlas_folder):
    ''' Computes CST load (weighted, using binarized atlas), see https://onlinelibrary.wiley.com/doi/10.1002/ana.24510
        Note: here, FSL atlas binarized at 70% is used

        Parameter
        ----------
        lesion : 3D array containing the mask of the overlap
        side: left (0) or right (1) hemicord

        Return
        -------
        weighted_overlap
    '''
    # ATLAS
    if side == 0:
        template = nib.load(atlas_folder + 'Atlas/FSL_CST_L_resized_bin_thr70.nii.gz')
    elif side == 1:
        template = nib.load(atlas_folder + 'Atlas/FSL_CST_R_resized_bin_thr70.nii.gz')
    template_np = template.get_fdata()
    sum_slice = template_np.sum(axis=(0, 1))  # Sum of voxels per slice (atlas)
    with np.errstate(all='ignore'):
        ratio_slice = sum_slice.max() / sum_slice  # Maximum number of voxels

    # LESION
    # Sum of voxel per slice (lesion)
    sum_slice_lesion = lesion.sum(axis=(0, 1))
    with np.errstate(all='ignore'):
        weighted_overlap = np.nansum(sum_slice_lesion * ratio_slice)
    return weighted_overlap
