import os.path
import os
from turtle import pd
from pyparsing import col
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import glob

class iCAPs:
    '''
    The iCAPs class is used to manipulate and visualize
    iCAPs extracted using the MATLAB pipeline (https://c4science.ch/source/iCAPs/)

    Attributes
    ----------
    config : dict
        Contains information regarding subjects, sessions, paths, etc.
    k : int
        Number of clusters (i.e., iCAPs)
    sorting method : str
        Describes the method used to sort iCAPs
            'no_sorting' = keep original order
            'rostrocaudal' = from ascending to descending iCAPs
    icap_data : 4D array
        Contains the z-scored iCAPs
    icap_order : array 
        Contains the indices to re-order iCAPs based on method defined by sorting_method
    spinal_levels : array
        Contains the spinal level corresponding to each iCAP (already re-ordered)
    '''
    
    def __init__(self, config, k, sorting_method='rostrocaudal'):
        self.config = config
        self.k = k
        self.sorting_method = sorting_method
        # Load iCAPs
        self.icap_data = nib.load(self.config['icap_root'] + self.config['icap_folder'] + 'K_' + str(self.k) + '_Dist_cosine_Folds_20/iCAPs_z.nii').get_fdata()        
        self.icap_order = self._sort_icaps(self.sorting_method)
        # Re-order iCAPs so that we do it only once
        self.icap_data = self.icap_data[:,:,:,self.icap_order]
        self.spinal_levels = self._match_levels()

    def plot(self, k_per_line=None, lthresh=2.3, uthresh=4.0, centering_method='max', show_spinal_levels=False, colormap='autumn', save_results=False):
        ''' Plot iCAPs overlaid on PAM50 template (coronal and axial views are shown)
        
        Inputs
        ----------
        k_per_line: str
            Number of iCAPs to display per line (default = will be set to total number of iCAPs)
        lthresh : float
            Lower threshold value to display z-scored iCAPs (default = 2.3)
        uthresh : float
            Upper threshold value to display z-scored iCAPs (default = 4.0)
        centering_method : str
            Method to center display in the anterio-posterior direction (default = 'max')
                'max' to center based on voxel with maximum activity
                'middle' to center in the middle of the volume
        show_spinal_levels : boolean
            Defines whether spinal levels are displayed or not (default = False)
        colormap : str
            Defines colormap used to plot iCAPs (default = 'autumn')
        save_results : boolean
            Set to True to save figure (default = False)'''

        # By default, use a single line for all iCAPs, otherwise use provided value
        if (k_per_line is not None and k_per_line <= self.k) or k_per_line is None:
            k_per_line = self.k if k_per_line is None else k_per_line
        else:
            raise(Exception('Number of iCAPs per line should be inferior or equal to the total number of iCAPs.'))

        # Load template image for background
        template_img = nib.load(self.config['template_path'])
        template_data = template_img.get_fdata()

        if show_spinal_levels == True: # Load levels if needed
            # Find list of spinal levels to consider (defined in config)
            levels_list = sorted(glob.glob(self.config['levels_path'] + 'spinal_level_*.nii.gz')) # Sorted is used to make sure files are listed from low to high number (i.e., rostro-caudally)
            levels_data = np.zeros((self.icap_data.shape[0],self.icap_data.shape[1],self.icap_data.shape[2],len(levels_list))) # To store spinal levels, based on size of iCAP data (corresponding to template) & number of spinal levels in template
            # Loop through levels & store data
            for lvl in range(0,len(levels_list)):
                level_img = nib.load(levels_list[lvl])
                levels_data[:,:,:,lvl] = level_img.get_fdata()
                # Mask level data to use as overlays
                levels_data = np.where(levels_data > 0, levels_data, np.nan)       
        # To mask iCAPs, values below threshold are replaced by NaN
        icap_masked = np.where(self.icap_data > lthresh, self.icap_data, np.nan)

        # Compute number of columns/rows and prepare subplots accordingly 
        total_rows = (self.k//k_per_line + 1)*2 if self.k > k_per_line else 2
        fig, axs = plt.subplots(nrows=total_rows,ncols=k_per_line,figsize=(2*k_per_line, 4*total_rows))
        plt.axis('off')

        for i in range(0,self.k):
            # Column is the same for coronal & axial views
            col = i%k_per_line
            # Draw coronal views
            row_coronal = 0 if i<k_per_line else (i//k_per_line-1)*2+2
            axs[row_coronal,col].axis('off')
#            axs[row_coronal,col].set_title('iCAP' + str(i+1),fontsize=18,pad=20)
            axs[row_coronal,col].set_title(('iCAP' + str(i+1) + '\n level ' + str(self.spinal_levels[i]+1) if show_spinal_levels else 'iCAP' + str(i+1)),fontsize=18,pad=15)
            if centering_method == 'middle':
                axs[row_coronal,col].imshow(np.rot90(template_data[:,70,:]),cmap='gray',origin='lower');
                if show_spinal_levels == True:
                    axs[row_coronal,col].imshow(np.rot90(levels_data[:,70,:,self.spinal_levels[i]]),cmap='gray')
                axs[row_coronal,col].imshow(np.rot90(icap_masked[:,template_data.shape[1]//2,:,i]),vmin=lthresh, vmax=uthresh,cmap=colormap)
            elif centering_method == 'max':
                max_y = int(np.where(icap_masked == np.nanmax(icap_masked[:,:,:,i]))[1])
                axs[row_coronal,col].imshow(np.rot90(template_data[:,max_y,:]),cmap='gray');
                if show_spinal_levels == True:
                    axs[row_coronal,col].imshow(np.rot90(levels_data[:,max_y,:,self.spinal_levels[i]]),cmap='gray')
                axs[row_coronal,col].imshow(np.rot90(icap_masked[:,max_y,:,i]),vmin=lthresh, vmax=uthresh,cmap=colormap)
            else:
                raise(Exception(f'{centering_method} is not a supported centering method.'))
            
            # Draw axial views
            row_axial = 1 if i<k_per_line else (i//k_per_line-1)*2+3
            axs[row_axial,col].axis('off');
            max_z = int(np.where(icap_masked == np.nanmax(icap_masked[:,:,:,i]))[2])
            axs[row_axial,col].imshow(template_data[:,:,max_z].T,cmap='gray');
            axs[row_axial,col].imshow(icap_masked[:,:,max_z,i].T,vmin=lthresh, vmax=uthresh,cmap=colormap)
            # To "zoom" on the spinal cord, we adapt the x and y lims
            axs[row_axial,col].set_xlim([icap_masked.shape[0]*0.2,icap_masked.shape[0]*0.8])
            axs[row_axial,col].set_ylim([icap_masked.shape[1]*0.2,icap_masked.shape[1]*0.8])
            axs[row_axial,col].set_anchor('N')

        # If option is set, save results as a png
        if save_results == True:
            plt.savefig(self.config['output_dir'] + self.config['tag_results'] + '_k_' + str(self.k) + '.png')

    def extract_durations(self):
        ''' Extract durations from xlsx file containing temporal characteristics
            /!\ First, need to extract xlxs file from mat file using MATLAB (tempChar2xls.m)
            Here, we focus on total & average duration
        
        Outputs
        ----------
        durations_df : df
            Contains total (as a percentage of the run duration) and average (as a number of volumes) iCAP duration (per subject, session and icap)  
        '''

        print('EXTRACT TEMPORAL CHARACTERISTICS')
        # First, check that file exists and load it in case
        temp_char_file = self.config['icap_root'] + self.config['icap_folder'] + 'K_'+ str(self.k) + '_Dist_cosine_Folds_20/' + self.config['temp_char_name'] + '.xlsx'
        if os.path.isfile(temp_char_file):
            print('... Opening xlsx file')
            temp_char = pd.read_excel(temp_char_file,sheet_name=None, header=None)            
        else:
            raise(Exception(f'{temp_char_file} does not exist. Has it been extracted from the .mat file?'))

        print('... Prepare subject info')
        sessions = []
        subs = []
        for sub in self.config['list_subjects']:
            subs.append(sub)
            sessions.append(self.config['list_subjects'][sub]['sess'][0])
        colnames = [subs,sessions]

        print('... Format durations into a dataframe')
        df_tot_dur = temp_char.get('duration_total_perc')
        df_avg_dur = temp_char.get('duration_avg_counts')

        # Re-order iCAPs
        df_tot_dur_reordered = df_tot_dur.copy()
        df_tot_dur_reordered = df_tot_dur_reordered.reindex(self.icap_order) # Changer order 
        df_tot_dur_reordered.reset_index(drop=True, inplace=True) # Reset indices
        
        df_avg_dur_reordered = df_avg_dur.copy()
        df_avg_dur_reordered = df_avg_dur_reordered.reindex(self.icap_order) # Changer order 
        df_avg_dur_reordered.reset_index(drop=True, inplace=True) # Reset indices
      
        df_tot_dur_formatted = self._format_dur_df(df_tot_dur_reordered,colnames,'total_dur')
        df_avg_dur_formatted = self._format_dur_df(df_avg_dur_reordered,colnames,'avg_dur')
        
        durations_df = df_tot_dur_formatted.merge(df_avg_dur_formatted['avg_dur'],left_index=True, right_index=True, how='outer')

        return durations_df

    def plot_durations(self, durations_df, grouped=True):
        ''' Plot average and total duration
        
        Inputs
        ----------
        durations_df : df
            Contains total (as a percentage of the run duration) and average (as a number of volumes) iCAP duration (per subject, session and icap)  
        grouped : boolean
            Plot each session/group independently (default: True)
        '''
        fig,axs = plt.subplots(1,2,figsize=(15,4));
        if grouped == True:
            sns.barplot(data=durations_df,x = "iCAPs", y="total_dur",hue="Session",ax=axs[0],palette='flare');
            sns.barplot(data=durations_df,x = "iCAPs", y="avg_dur",hue="Session",ax=axs[1],palette='flare');
        else:
            sns.barplot(data=durations_df,x = "iCAPs", y="total_dur",ax=axs[0],palette='flare');
            sns.barplot(data=durations_df,x = "iCAPs", y="avg_dur",ax=axs[1],palette='flare');
        axs[0].set_title('Total duration'); axs[0].set_ylabel('% of total run duration');
        axs[1].set_title('Average duration'); axs[1].set_ylabel('# of volumes');
        # Place legend in bottom left of plot
        axs[0].legend(loc='lower left')
        axs[1].legend(loc='lower left')

    def _sort_icaps(self, sorting_method):
        ''' Sort iCAPs based on sorting_method (e.g., rostrocaudally)
        
        Inputs
        ----------
        sorting_method : str
            Method used to sort iCAPs (e.g., 'rostrocaudal')

        Outputs
        ----------
        sort_index : list
            Contains the indices of the sorted iCAPs   
        '''    
        if sorting_method == 'rostrocaudal':
            max_z = []
            for i in range(0,self.k):
                max_z.append(int(np.where(self.icap_data == np.nanmax(self.icap_data[:,:,:,i]))[2]))
            sort_index = np.argsort(max_z)
            sort_index = sort_index[::-1] # Invert direction to go from up to low
        elif sorting_method == 'no_sorting':
            sort_index = list(range(self.k))
        else:
            raise(Exception(f'{sorting_method} is not a supported sorting method.'))
        return sort_index

    def _match_levels(self):
        ''' Match iCAPs to corresponding spinal levels
        Output
        ----------
        spinal_levels : list
            Array containing one value per iCAP 
                C1 = 1, C2 = 2, C3 = 3, C4 = 4, etc.
        '''
        
        # Find list of spinal levels to consider (defined in config)
        levels_list = sorted(glob.glob(self.config['levels_path'] + 'spinal_level_*.nii.gz')) # Sorted is used to make sure files are listed from low to high number (i.e., rostro-caudally)
        
        # Prepare structures
        levels_data = np.zeros((self.icap_data.shape[0],self.icap_data.shape[1],self.icap_data.shape[2],len(levels_list))) # To store spinal levels, based on size of iCAP data (corresponding to template) & number of spinal levels in template
        spinal_levels = np.zeros(self.k,dtype='int') # To store corresponding spinal levels

        # Loop through levels & store data
        for lvl in range(0,len(levels_list)):
            level_img = nib.load(levels_list[lvl])
            levels_data[:,:,:,lvl] = level_img.get_fdata()
        # For each iCAP, find rostrocaudal position of point with maximum intensity
        max_intensity = np.zeros(self.icap_data.shape[3],dtype='int')
        for i in range(0,self.k):
            max_intensity[i] = np.where(self.icap_data == np.nanmax(self.icap_data[:,:,:,i]))[2]
            # Take this point for each level (we focus on rostrocaudal position and take center of FOV for the other dimensions)
            level_vals = levels_data[levels_data.shape[0]//2,levels_data.shape[1]//2,max_intensity[i],:]
            spinal_levels[i] = np.argsort(level_vals)[-1] if np.sum(level_vals) !=0 else -1 # Take level with maximum values (if no match, use -1)
    
        return spinal_levels

    def _format_dur_df(self,df_to_format,colnames,metric):
        ''' Format dataframe containing durations
            1. Add column names (subjects and sessions)
            2. Add column with iCAP numbers
            3. From columns to rows
        
        Inputs
        ----------
        df_to_format : df
            Dataframe, as extracted from the xlsx file
        colnames : list
            Contains subjects and sessions, to put as column names
        metric : str
            Name of the metric of interest in this df file

        Outputs
        ----------
        formatted_df : df
            Dataframe, formatted to ease analyses
        '''
        formatted_df = df_to_format.copy()
        formatted_df.columns = pd.MultiIndex.from_arrays(colnames) # Add double rows of column names (subjects and sessions)
        formatted_df['iCAPs'] = formatted_df.index+1 # Add +1 to have iCAPs from 1 to k
        formatted_df = formatted_df.melt(id_vars='iCAPs', var_name=['Subject','Session'], value_name=metric) # Reorganize to have everything as rows

        return formatted_df