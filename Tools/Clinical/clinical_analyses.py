import numpy as np
import pandas as pd
import os.path
import nibabel as nib
import pingouin as pg
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

class ClinicalAnalyses:
    '''
    The ClinicalAnalyses class is used to explore clinical-related metrics

    Attributes
    ----------
    config : dict
        Contains information regarding subjects, sessions, masks, etc.
    '''

    def __init__(self, config):
        self.config = config
        print(f'Creating instance for config {self.config["tag_results"]}')
        print(f'overwrite_clinical: {self.config["overwrite_clinical"]}')

    def get_patient_info(self):
        '''Extract information on lesions and put them into a dataframe

        Returns
        ----------
        clinical_infos : df
            Contains patient-related infos
            - weighted CST load (L/R)
            - total lesion volume 
            - clinical scores
            - demographics (age, gender)
        '''   
         # We just run everything is this has not been done already or if we want to start from scratch
        if (not os.path.isfile(self.config['output_dir'] + 'clinical_infos_' + self.config['tag_results'] + '.pkl')) or self.config['overwrite_clinical']:
            print(f'Extracting clinical data...')
            subs = []
            sessions = []
            weighted_load_L = []
            weighted_load_R = []
            lesion_vol = []
            pinch = []
            mas = []
            fm = []
            age = []
            gender = []
            
            # Load clinical file
            raw_clinical_data = pd.read_excel(self.config['clinical_file'],sheet_name='ClinicalScores')

            # Loop through patients only
            patients_only = (sub for sub in self.config['list_subjects'] if self.config['list_subjects'][sub]['subtype'] == 'P')
            for sub in patients_only:  
                for session in self.config['list_subjects'][sub]['sess']:
                    subs.append(sub)
                    sessions.append(session)
                    # Get lesion info
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
                    weighted_load = weighted_load_L + weighted_load_R
                    # Add also total volume of lesion
                    if self.config['acute_only'] == True and os.path.isfile(self.config['lesion_dir'] + 'Lesions/' + sub + '-' + session + '-acute_only-overlap_FSL_CST_' + side + '.nii.gz'):
                        lesion = nib.load(self.config['lesion_dir'] + 'Lesions/' + sub + '-' + session + '-lesion_acute_only.nii')
                    else:
                        lesion = nib.load(self.config['lesion_dir'] + 'Lesions/' + sub + '-' + session + '-lesion.nii')
                    lesion_np = lesion.get_fdata()
                    lesion_vol.append(np.sum(lesion_np))
                    del tmp_load

                    # To map session names with indices
                    sessions_ix = {'T2': 0, 'T3': 1, 'T4': 2}

                    # Get clinical info
                    # 1. Fugl-Meyer (corrected)
                    # Column 23-25
                    fm.append(raw_clinical_data.loc[raw_clinical_data.iloc[:,1]==sub, raw_clinical_data.columns[23+sessions_ix.get(session)]].values[0])
                    # 2. ARAT pinch
                    # Column 3/5/7
                    pinch.append(raw_clinical_data.loc[raw_clinical_data.iloc[:,1]==sub, raw_clinical_data.columns[3+2*sessions_ix.get(session)]].values[0])
                    # 3. MAS
                    # Column 9/11/13
                    mas.append(raw_clinical_data.loc[raw_clinical_data.iloc[:,1]==sub, raw_clinical_data.columns[9+2*sessions_ix.get(session)]].values[0])
                    # 4. Age
                    age.append(raw_clinical_data.loc[raw_clinical_data.iloc[:,1]==sub, raw_clinical_data.columns[18]].values[0])
                    # 5. Gender
                    gender.append(raw_clinical_data.loc[raw_clinical_data.iloc[:,1]==sub, raw_clinical_data.columns[19]].values[0])
            
            colnames = ["sub", "sess", "CST_L", "CST_R", "CST","vol", "fm", "pinch", "mas", "age", "gender"]
            clinical_infos = pd.DataFrame(list(zip(subs, sessions, weighted_load_L, weighted_load_R, weighted_load, lesion_vol, fm, pinch, mas, age, gender)), columns=colnames)
            clinical_infos.to_pickle(self.config['output_dir'] + 'clinical_infos_' + self.config['tag_results'] + '.pkl')  # Save dataframe

        else:
            print('Clinical information already extracted, loading from .pkl fike...')
            clinical_infos = pd.read_pickle(self.config['output_dir'] + 'clinical_infos_' + self.config['tag_results'] + '.pkl')
        
        print(f'Done!')
        return clinical_infos

    def compute_correlations(self, clinical_data, clinical_names, totest_values, totest_names, rm_confounds=True, plot=True):
        '''Compute correlation between clinical metrics and values of interest 

        Inputs
        ----------
        clinical_data : df
            Contains patient-related data
        clinical_names : list of str
            Contains names of clinical metrics to plot (e.g., ['fm','mas'])
        totest_values : array
            Contains values of interest (e.g., couplings between specific rois, icap metrics, etc.)
        totest_names : list of str
            Contains names of values of interest (size should be equal to the number of columns in totest_values)
        rm_confounds : boolean
            Remove confounds when computing correlations if True (default)
        plot : boolean
            Plot correlation matrices if True (default)
        ''' 
        if totest_values.shape[1] == len(totest_names):
            if totest_values.shape[0] == clinical_data.shape[0]:
                clinical_totest = clinical_data.copy() # Copy clinical dataframe
                for totest_ix, totest in enumerate(range(0, len(totest_names))):
                    clinical_totest[totest_names[totest_ix]] = totest_values[:,totest_ix] # Add metrics of interest to new dataframes
            else:
                raise Exception('The number of subjects should be the same for the clinical scores and the metrics to test')
        else:
            raise Exception('The number of columns of the array to test should be equal to the number of names passed')
        
        # Partial correlations to remove confounds
        tested = []
        clinics = []
        rhos = []
        ps = []
        for clinical in clinical_names:
            for test in totest_names:
                tested.append(test)
                clinics.append(clinical)
                if rm_confounds==True:
                    pcorr = pg.partial_corr(data=clinical_totest, x=test, y=clinical, covar=['age','gender'])
                    rhos.append(pcorr.values[0][1])
                    ps.append(pcorr.values[0][3])
                else:
                    corr = pearsonr(clinical_totest[test], clinical_totest[clinical])
                    rhos.append(corr[0])
                    ps.append(corr[1])

        colnames = ["tested","clinical","rho","p"]
        clinical_correlations = pd.DataFrame(list(zip(tested,clinics,rhos,ps)), columns=colnames)
          
        if plot == True:
            fig, axes = plt.subplots(len(totest_names),len(clinical_names), figsize=(15,5), sharey=True)
            fig.suptitle('Correlations with clinical metrics',y=1,fontsize='x-large',weight="bold");
            for testix,test in enumerate(totest_names):
                for clinicalix,clinical in enumerate(clinical_names):
                    sns.scatterplot(ax=axes[testix,clinicalix],data=clinical_totest,x=clinical,y=test);
            fig.savefig(self.config['output_dir'] + 'plots/clinical_corrs_' + self.config['tag_results'] + '.pdf')

        return clinical_correlations

    def plot_info(self, clinical_data, clinical_names):
        '''Plot clinical metrics 

        Inputs
        ----------
        clinical_data : df
            Contains patient-related data
        clinical_names : list of str
            Contains names of clinical metrics to plot (e.g., ['fm','mas'])
        ''' 
        # First, define number of subplots based on values to plot
        fig, axes = plt.subplots(1,len(clinical_names), figsize=(10,4), sharey=False)
        fig.suptitle('Clinical metrics',y=1,fontsize='x-large',weight="bold");
        for metric_ix,metric in enumerate(clinical_names):
            sns.swarmplot(ax=axes[metric_ix],data=clinical_data,x='sess',y=metric,size=6,palette="rocket");
        fig.tight_layout()
        # Save figure as pdf
        fig.savefig(self.config['output_dir'] + 'plots/clinical_infos_' + self.config['tag_results'] + '.pdf')

# Utilities

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
