from sqlite3 import converters
import numpy as np
import pandas as pd
from scipy.signal import filtfilt, cheby2
import nibabel as nib
import os.path
import os
from nilearn.masking import apply_mask
from joblib import Parallel, delayed
import time
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp
from scipy.stats import mannwhitneyu
from scipy.stats import permutation_test
from statsmodels.stats.multitest import fdrcorrection
from nipype.interfaces.fsl.maths import StdImage
from nipype.interfaces.fsl import ImageStats
from scipy import stats
from nilearn import image

class StaticFC:
    '''
    The StaticFC class is used to compute and visualize sFC
    metrics on functional timecourses
        - connectivity analyses
        - ALFF analyses

    Attributes
    ----------
    config : dict
        Contains information regarding subjects, sessions, rois, etc.
    '''

    def __init__(self, config):
        self.config = config
        print(f'Creating instance for config {self.config["tag_results"]}')
        print(f'overwrite_bp: {self.config["overwrite_bp"]}')
        print(f'overwrite_tc: {self.config["overwrite_tc"]}')
        print(f'overwrite_corr: {self.config["overwrite_corr"]}')
        print(f'overwrite_smooth: {self.config["overwrite_smooth"]}')
        print(f'overwrite_alff_maps: {self.config["overwrite_alff_maps"]}')
        print(f'overwrite_alff_rois: {self.config["overwrite_alff_rois"]}')

    def prepare_data(self):
        ''' Prepare the data
        1.  Apply a bandpass filter on the denoised images
            Note: filter choice is based on Barry et al. 2014, eLife
        2.  Normalize to PAM50 (nn interpolation)
        3.  Flip normalized images if neeeded
        4.  Smoothing

        Outputs
        ----------
        denoised_img_bp.nii.gz
            Filtered image
        denoised_img_bp_pam50_nn.nii.gz
            Filtered image normalized to the PAM50 template
        denoised_img_bp_pam50_nn_flipped.nii.gz
            Filtered image normalized to the PAM50 template, flipped (left <-> right)
            /!\ This applies only to patients, if they have lesions on the left side
        '''

        print('PREPARE DATA')

        # Define filter (same type of filter as Barry 2014)
        b, a = cheby2(4, 30, self.config['bp_range'],
                      'bandpass', fs=1./self.config['TR'])

        # Linearize list of subjects & sessions to run in parallel
        all_sub = []  # This will contain all the paths without extension, so that they suffixes can be added later on
        sub_to_flip = [] # This will contain paths only for patients where flipping is required
        for sub in self.config['list_subjects']:
            for sess in self.config['list_subjects'][sub]['sess']:
                #  For patients, session name is included in paths, files, etc.
                if self.config['list_subjects'][sub]['subtype'] == 'P':
                    sub_data_path = self.config['root']['P'] + sub + '/' + sess + self.config['func_dir']['P']
                    sub_func_file = sub + '-' + sess + '-' + self.config['func_name']['P']
                    # If lesion on the left side
                    if self.config['list_subjects'][sub]['side'] == 'L': 
                        sub_to_flip.append(sub_data_path+sub_func_file)
                    else: 
                        pass
                elif self.config['list_subjects'][sub]['subtype'] == 'H':
                    sub_data_path = self.config['root']['H'] + sub + self.config['func_dir']['H'] + sess + '/'
                    sub_func_file = sub + '_' + self.config['func_name']['H']
                else:
                    raise Exception(
                        f'Unknown subtype {self.config["list_subjects"][sub]["subtype"]}. Should be H or P')
                all_sub.append(sub_data_path+sub_func_file)

        print(f'Filtering data in the range [{self.config["bp_range"][0]}-{self.config["bp_range"][1]}] Hz')
       
        print('... Apply filter')
        print(f'Overwrite old files: {self.config["overwrite_bp"]}')
        start = time.time()
        
        Parallel(n_jobs=self.config['n_jobs'],
                 verbose=100,
                 backend='loky')(delayed(self._apply_filter)(sub, b, a)
                                   for sub in all_sub)
        print("... Operation performed in %.3f s" % (time.time() - start))

        print('... Normalize filtered image')
        print(f'Overwrite old files: {self.config["overwrite_bp"]}')
        start = time.time()
        Parallel(n_jobs=self.config['n_jobs'],
                 verbose=100,
                 backend='loky')(delayed(self._apply_norm)(sub,interp='nn')\
                 for sub in all_sub)
        print("... Operation performed in %.3f s" % (time.time() - start))

        print('... Flip normalized filtered image')
        print(f'Overwrite old files: {self.config["overwrite_bp"]}')
        start = time.time()
        Parallel(n_jobs=self.config['n_jobs'],
                 verbose=100,
                 backend='loky')(delayed(self._flip)(sub)\
                 for sub in sub_to_flip)      
        print("... Operation performed in %.3f s" % (time.time() - start))
    
        print('...DONE!')

    def extract_tcs(self):
        '''Extract timecourses within specific rois and put them into a dataframe

        Returns
        ----------
        tcs : df
            Contains timecourses within each roi
            Note: also saved as a .pkl file
        '''

        print('Timecourses extraction')
        print(f'Overwrite old files: {self.config["overwrite_tc"]}')

        # We just run everything is this has not been done already or if we want to start from scratch
        if not os.path.isfile(self.config['output_dir'] + 'tcs_' + self.config['tag_results'] + '.pkl') or self.config['overwrite_tc']:
            t = np.empty((0,), int)  # Timestamps
            tcs = np.empty((0,), float)  # Timecourses
            subs = []
            sessions = []
            rois = []

            for sub in self.config['list_subjects']:
                print(f'Extracting timecourses for subject {sub}')
                for sess in self.config['list_subjects'][sub]['sess']:
                    print(f'...Session {sess}')
                    #  For patients, session name is included in paths, files, etc.
                    if self.config['list_subjects'][sub]['subtype'] == 'P':
                        sub_data_path = self.config['root']['P'] + sub + '/' + sess + self.config['func_dir']['P']
                        if self.config['list_subjects'][sub]['side'] == 'R':
                            if self.config['sub_prefix']['P'] == True: # File name different if prefix with subject name or not
                                sub_func_file = sub + '-' + sess + '-' + self.config['func_name']['P'] + '_bp_pam50_nn.nii.gz' # Name of the normalized filtered fMRI
                            else:
                                sub_func_file = self.config['func_name']['P'] + '_bp_pam50_nn.nii.gz' # Name of the normalized filtered fMRI
                        elif self.config['list_subjects'][sub]['side'] == 'L':
                            if self.config['sub_prefix']['P'] == True: # File name different if prefix with subject name or not
                                sub_func_file = sub + '-' + sess + '-' + self.config['func_name']['P'] + '_bp_pam50_nn_flipped.nii.gz' # Name of the normalized filtered fMRI flipped!
                            else:
                                sub_func_file = self.config['func_name']['P'] + '_bp_pam50_nn_flipped.nii.gz' # Name of the normalized filtered fMRI flipped!

                        else:
                            raise Exception(
                                f'Unknown side {self.config["list_subjects"][sub]["side"]}. Should be L or R') 
                    elif self.config['list_subjects'][sub]['subtype'] == 'H':
                        sub_data_path = self.config['root']['H'] + sub + self.config['func_dir']['H'] + sess + '/'
                        if self.config['sub_prefix']['H'] == True: # File name different if prefix with subject name or not
                            # Name of the normalized filtered fMRI
                            sub_func_file = sub + '_' + self.config['func_name']['H'] + '_bp_pam50_nn.nii.gz'
                        else: 
                            sub_func_file = self.config['func_name']['H'] + '_bp_pam50_nn.nii.gz'
                    else:
                        raise Exception(
                            f'Unknown subtype {self.config["list_subjects"][sub]["subtype"]}. Should be H or P')
                    # Then, we take the average signal in each region of interest
                    for roi in self.config['roi_names']:
                        print(f'......In {roi}')
                        tc = apply_mask(
                            sub_data_path+sub_func_file, self.config['template_path'] + '/masks/' + roi + '.nii.gz')
                        mean_tc = np.mean(tc, axis=1)

                        # Then, prepare data to include in dataframe
                        for tval in range(0, len(mean_tc)):
                            t = np.append(t, tval)
                            subs.append(sub)
                            sessions.append(sess)
                            rois.append(roi)
                        tcs = np.append(tcs, mean_tc)

            colnames = ["t", "tc", "sub", "sess", "roi"]
            tcs = pd.DataFrame(list(zip(t, tcs, subs, sessions, rois)), columns=colnames)
            tcs.to_pickle(self.config['output_dir'] + 'tcs_' + self.config['tag_results'] + '.pkl')  # Save dataframe
        else:
            print('Timecourses already extracted, loading from .pkl fike...')
            tcs = pd.read_pickle(self.config['output_dir'] + 'tcs_' + self.config['tag_results'] + '.pkl')

        return tcs

    def compute_correlations(self, tcs, rois=None, plot=True):
        '''Compute correlation matrices for specific rois and put them into a dataframe

        Inputs
        ----------
        tcs : df
            Dataframe containing all timecourses 
        rois : list
            List of rois between which correlations should be computed (default = rois from config file)
        plot: boolean
            Set to True to plot average correlation matrices
        
        Returns
        ----------
        corrs : df
            Contains Z-transformed correlations for all subjects, sessions and roi pairs 
        corrs_stats : df
            Contains statistical significance (p values) for all sessions and roi pairs 
        '''

        print(f'ROI-TO-ROI CORRELATION')
        print(f'Overwrite old files: {self.config["overwrite_corr"]}')

        # By default, use all roi masks in configuration file, otherwise indicate lst as argument
        rois = self.config['roi_names'] if rois is None else rois
        # We just run everything is this has not been done already or if we want to start from scratch
        if not os.path.isfile(self.config['output_dir'] + 'corrs_' + self.config['tag_results'] + '.pkl') or self.config['overwrite_corr']:
            print(f'...Compute correlations')
            
            # Initialize structures
            rhos_Z = []
            subs = [] 
            sessions = []
            rois1 = []
            rois2 = []
            
            for sub in self.config['list_subjects']:
                for sess in self.config['list_subjects'][sub]['sess']:
                    for roi1 in rois:
                        for roi2 in rois:
                            # Here we take the Z transform of the correlation coefficient
                            rhos_Z.append(np.arctanh(np.corrcoef(tcs.loc[(tcs['sub']==sub) & (tcs['sess']==sess) & (tcs['roi']==roi1),'tc'].values, tcs.loc[(tcs['sub']==sub) & (tcs['sess']==sess) & (tcs['roi']==roi2),'tc'].values)[1,0]))
                            rois1.append(roi1)
                            rois2.append(roi2)
                            subs.append(sub)
                            sessions.append(sess)
            colnames = ["sub","sess","roi1","roi2","rho_Z"]
            corrs = pd.DataFrame(list(zip(subs, sessions, rois1, rois2, rhos_Z)), columns=colnames)
            corrs.to_pickle(self.config['output_dir'] + 'corrs_' + self.config['tag_results'] + '.pkl')  # Save dataframe 
            
            print(f'...Compute statistical significance')
            
            sessions = [] # Reinitialize all values to build stats dataframe
            rois1 = []
            rois2 = []
            pvals = []
            pvals_corr = []
            for sess in corrs['sess'].unique().tolist():
                for roi1 in rois:
                    for roi2 in rois:
                        rois1.append(roi1)
                        rois2.append(roi2)
                        sessions.append(sess)
                        pvals.append(mannwhitneyu(corrs[(corrs['sess']==sess) & (corrs['roi1']==roi1) & (corrs['roi2']==roi2)]['rho_Z'].values,0).pvalue)        
                        #pvals.append(permutation_test(corrs[(corrs['sess']==sess) & (corrs['roi1']==roi1) & (corrs['roi2']==roi2)]['rho_Z'].values,0).pvalue)        
                # FDR correction of p values for a particular session (Benjamini-Hochberg)
                sess_indices  = [index for (index, item) in enumerate(sessions) if item == sess] # Take all the indices for this session
                ptocorrect = np.reshape([pvals[index] for index in sess_indices],(len(rois),len(rois))) # Reshape as correlation matrix
                pcorr = fdrcorrection(ptocorrect[np.triu_indices(len(rois),k=1)])[1] # Compute FDR correction on triangular matrix only
                pcorr_tri = np.zeros((len(rois), len(rois))) # Prepare empty triangular matrix (i.e., to have right structure for dataframe)
                pcorr_tri[np.triu_indices(len(rois), k=1)] = pcorr # Fill it with corrected p values
                pvals_corr.extend(pcorr_tri.flatten()) # Flatten to create column
                
            colnames = ["sess","roi1","roi2","p","p_fdr"]
            corrs_stats = pd.DataFrame(list(zip(sessions, rois1, rois2, pvals, pvals_corr)), columns=colnames)
            corrs_stats.to_pickle(self.config['output_dir'] + 'corrs_stats_' + self.config['tag_results'] + '.pkl')  # Save dataframe
            
        else:
            print('Correlations already computed, loading from .pkl fike...')
            corrs = pd.read_pickle(self.config['output_dir'] + 'corrs_' + self.config['tag_results'] + '.pkl')
            corrs_stats = pd.read_pickle(self.config['output_dir'] + 'corrs_stats_' + self.config['tag_results'] + '.pkl')
        
        if plot == True:
            fig, axes = plt.subplots(1,len(corrs['sess'].unique().tolist()), figsize=(15,5), sharey=True, squeeze=False)
            fig.suptitle('Mean correlations',y=0.94,fontsize='x-large',weight="bold");
            # Generate a mask for the lower triangle
            mask_lt = np.triu(np.ones_like(rois, dtype=bool))
            mask_ut = np.tril(np.ones_like(rois, dtype=bool))
            for sessix,sess in enumerate(corrs['sess'].unique().tolist()):
                sns.heatmap(np.reshape(corrs[corrs['sess']==sess].groupby(['roi1','roi2'],sort=False)['rho_Z'].mean().values,(len(rois),len(rois))),
                            ax=axes.flat[sessix], linewidths=.5,vmin=-0.2,vmax=0.2,square=True,cmap=sns.diverging_palette(220, 20, as_cmap=True),mask=mask_lt,cbar_kws={'label': 'Fisher Z','shrink':0.9})
                sns.heatmap(np.reshape(corrs_stats[corrs_stats['sess']==sess]['p_fdr'].values,(len(rois),len(rois))),
                            xticklabels=rois,
                            yticklabels=rois,
                            ax=axes.flat[sessix],linewidths=.5,vmin=0,vmax=0.05,square=True,cmap="Greys_r",mask=mask_ut,cbar_kws={'label': 'Corrected p-value','shrink':0.9})        
                axes.flat[sessix].set_title(sess)
            fig.savefig(self.config['output_dir'] + 'plots/corrs_' + self.config['tag_results'] + '.pdf')

        return corrs, corrs_stats

    def compare_correlations(self, corrs, groups_to_compare, paired=False):
        '''Compare correlation matrices edge by edge

        Inputs
        ----------
        corrs : df
            Dataframe containing all correlations
        groups_to_compare : list
            List the two groups between which comparison should be done
        paired: boolean
            Set to True if subjects between groups are paired (default: False)
        
        Returns
        ----------
        corr_diff : array
            Matrix of the difference of correlations between group 1 and group 2
        comp_stats : array
            Contains p values for all edges
        '''

        print(f'COMPARE FC PATTERNS BETWEEN GROUPS')
        
        # Check if two groups to compare
        if len(groups_to_compare) != 2:
            raise Exception('Two groups need to be indicated for the comparison!')
        else:
            print(f'Group 1: {groups_to_compare[0]}')
            print(f'Group 2: {groups_to_compare[1]}')
        
        pvals = []
        rois1 = []
        rois2 = []

        for roi1 in self.config['roi_names']:
            for roi2 in self.config['roi_names']:
                g1 = corrs[(corrs['sess']==groups_to_compare[0]) & (corrs['roi1']==roi1) & (corrs['roi2']==roi2)]['rho_Z'].values
                g2 = corrs[(corrs['sess']==groups_to_compare[1]) & (corrs['roi1']==roi1) & (corrs['roi2']==roi2)]['rho_Z'].values
                rois1.append(roi1)
                rois2.append(roi2)
                if paired:
                    pvals.append(stats.ttest_rel(g1,g2).pvalue)
                else:
                    pvals.append(stats.ttest_ind(g1,g2).pvalue)
        
        colnames = ["roi1","roi2","p"]
        comp_stats = pd.DataFrame(list(zip(rois1, rois2, pvals)), columns=colnames)
           
        # Generate a mask for the lower triangle
        mask_lt = np.triu(np.ones_like(self.config['roi_names'], dtype=bool))
        mask_ut = np.tril(np.ones_like(self.config['roi_names'], dtype=bool))

        if paired:
            #for sub in [corrs['sess']==groups_to_compare[0]]['sub'].unique(): # Compute the difference per subject
            print('Todo')
        else:   
            corr_diff = np.reshape(corrs[corrs['sess']==groups_to_compare[0]].groupby(['roi1','roi2'],sort=False)['rho_Z'].mean().values,(len(self.config['roi_names']),len(self.config['roi_names']))) - np.reshape(corrs[corrs['sess']==groups_to_compare[1]].groupby(['roi1','roi2'],sort=False)['rho_Z'].mean().values,(len(self.config['roi_names']),len(self.config['roi_names'])))
            sns.heatmap(corr_diff, linewidths=.5,vmin=-0.2,vmax=0.2,square=True,cmap=sns.diverging_palette(220, 20, as_cmap=True),mask=mask_lt, cbar_kws={'label': 'Difference in Fisher Z','shrink':0.9})
            sns.heatmap(np.reshape(comp_stats['p'].values,(len(self.config['roi_names']),len(self.config['roi_names']))),
                            xticklabels=self.config['roi_names'],
                            yticklabels=self.config['roi_names'],
                            linewidths=.5,vmin=0,vmax=0.05,square=True,cmap="Greys_r",mask=mask_ut,cbar_kws={'label': 'p-value','shrink':0.9})        
            plt.savefig(self.config['output_dir'] + 'plots/corrs_diff_' + self.config['tag_results'] + '.pdf')
    
        return comp_stats, corr_diff

    def compute_alff_maps(self):
        '''Computes amplitude of low frequency fluctuations (ALFF) for each voxel
        (i.e., sum of the amplitudes in the low frequency band, as defined in config file)
        /!\ Flipping has been taken into account for patients!

        Inputs
        ----------
        denoised_img_bp_pam50_nn.nii.gz
            Filtered image normalized to the PAM50 template
        denoised_img_bp_pam50_nn_flipped.nii.gz
            Filtered image normalized to the PAM50 template, flipped (left <-> right)

        Outputs
        ----------
        denoised_img_alff_pam50.nii.gz
            ALFF image (i.e., std)
        denoised_img_alff_Z_pam50.nii.gz
            Z-scored ALFF image 

        '''
        print('ALFF MAPS COMPUTATION')
        print(f'Overwrite old files: {self.config["overwrite_alff_maps"]}')

        # Linearize list of subjects & sessions to run in parallel
        all_sub = []  # This will contain all the paths without extension, so that they suffixes can be added later on -> only for subjects that do not need to be flipped
        all_sub_flipped = [] # This will contain paths only for patients where flipping is required
        for sub in self.config['list_subjects']:
            for sess in self.config['list_subjects'][sub]['sess']:
                #  For patients, session name is included in paths, files, etc.
                if self.config['list_subjects'][sub]['subtype'] == 'P':
                    sub_data_path = self.config['root']['P'] + sub + '/' + sess + self.config['func_dir']['P']
                    sub_func_file = sub + '-' + sess + '-' + self.config['func_name']['P']
                    # If lesion on the left side
                    if self.config['list_subjects'][sub]['side'] == 'L': 
                        all_sub_flipped.append(sub_data_path+sub_func_file) # Add to list of subjects flipped
                    elif self.config['list_subjects'][sub]['side'] == 'R': 
                        all_sub.append(sub_data_path+sub_func_file) # Add to of subjects not flipped
                    else:
                        raise Exception(
                            f'Unknown side type {self.config["list_subjects"][sub]["side"]}. Should be L or R')
                elif self.config['list_subjects'][sub]['subtype'] == 'H':
                    sub_data_path = self.config['root']['H'] + sub + self.config['func_dir']['H'] + sess + '/'
                    sub_func_file = sub + '_' + self.config['func_name']['H']
                    all_sub.append(sub_data_path+sub_func_file) # Add to of subjects not flipped
                else:
                    raise Exception(
                        f'Unknown subtype {self.config["list_subjects"][sub]["subtype"]}. Should be H or P')
        
        print('... Smooth normalized filtered image that do not need to be flipped')
        print(f'Overwrite old files: {self.config["overwrite_smooth"]}')
        start = time.time()
        Parallel(n_jobs=self.config['n_jobs'],
                 verbose=100,
                 backend='loky')(delayed(self._smooth)(sub,False)
                                    for sub in all_sub)      
        print("... Operation performed in %.3f s" % (time.time() - start))

        print('... Smooth normalized filtered image that need to be flipped')
        print(f'Overwrite old files: {self.config["overwrite_smooth"]}')
        start = time.time()
        Parallel(n_jobs=self.config['n_jobs'],
                 verbose=100,
                 backend='loky')(delayed(self._smooth)(sub,True)
                                    for sub in all_sub_flipped)      
        print("... Operation performed in %.3f s" % (time.time() - start))

        print('... Running on data that do not need to be flipped')
        start = time.time()
        Parallel(n_jobs=self.config['n_jobs'],
                 verbose=100,
                 backend='loky')(delayed(self._alff)(sub,False)
                                   for sub in all_sub)
        print("... Operation performed in %.3f s" % (time.time() - start))

        print('... Running on data that need to be flipped')
        start = time.time()
        Parallel(n_jobs=self.config['n_jobs'],
                 verbose=100,
                 backend='loky')(delayed(self._alff)(sub,True)
                                   for sub in all_sub_flipped)
        print("... Operation performed in %.3f s" % (time.time() - start))

        print('...DONE!')

    def compute_alff_rois(self, rois_to_use=None, plot=True):
        '''Compute mean ALFF for specific rois and put them into a dataframe

        Needed
        ----------
        denoised_img_alff_Z_pam50.nii.gz 
            Z-scored ALFF image 
        
        Inputs
        ----------
        rois_to_use : list
            List of rois in which ALFF should be computed (default = rois from config file)
        plot: boolean
            Set to True to plot average ALFF
        
        Returns
        ----------
        alff : df
            Contains average ALFF for all rois
        '''

        print(f'ALFF IN ROIS')
        print(f'Overwrite old files: {self.config["overwrite_alff_rois"]}')

        # By default, use all roi masks in configuration file, otherwise indicate lst as argument
        rois_to_use = self.config['roi_names'] if rois_to_use is None else rois_to_use

        # We just run everything is this has not been done already or if we want to start from scratch
        if not os.path.isfile(self.config['output_dir'] + 'alffs_' + self.config['tag_results'] + '.pkl') or self.config['overwrite_alff_rois']:
            # Initialize structures
            alffs = []
            subs = [] 
            sessions = []
            rois = []
            
            for sub in self.config['list_subjects']:
                 print(f'Extracting ALFF for subject {sub}')
                 for sess in self.config['list_subjects'][sub]['sess']:
                    print(f'...Session {sess}')
                    for roi in rois_to_use:
                        subs.append(sub)
                        sessions.append(sess)
                        rois.append(roi)
                        #  For patients, session name is included in paths, files, etc.
                        if self.config['list_subjects'][sub]['subtype'] == 'P':
                            sub_data_path = self.config['root']['P'] + sub + '/' + sess + self.config['func_dir']['P']
                            sub_func_file = sub + '-' + sess + '-' + self.config['func_name']['P'] + '_alff_Z_pam50.nii.gz' # Name of the Z-score ALFF map
                        elif self.config['list_subjects'][sub]['subtype'] == 'H':
                            sub_data_path = self.config['root']['H'] + sub + self.config['func_dir']['H'] + sess + '/'
                            sub_func_file = sub + '_' + self.config['func_name']['H'] + '_alff_Z_pam50.nii.gz'
                            
                        else:
                            raise Exception(
                                f'Unknown subtype {self.config["list_subjects"][sub]["subtype"]}. Should be H or P')

                        # Computation of non-zero mean in ROI 
                        stats = ImageStats(in_file=sub_data_path + sub_func_file , mask_file=self.config['template_path'] + '/masks/' + roi + '.nii.gz',op_string= '-k %s -M');
                        alffs.append(stats.run().outputs.out_stat);
            
            colnames = ["sub","sess","roi","alff"]
            alffs = pd.DataFrame(list(zip(subs, sessions, rois, alffs)), columns=colnames)
            alffs.to_pickle(self.config['output_dir'] + 'alffs_' + self.config['tag_results'] + '.pkl')  # Save dataframe 
              
        else:
            print('ALFFs already extracted, loading from .pkl fike...')
            alffs= pd.read_pickle(self.config['output_dir'] + 'alffs_' + self.config['tag_results'] + '.pkl')
        
        if plot == True:
            plt.figure(figsize=(10,6))
            sns.barplot(x="roi",y="alff",hue="sess",data=alffs,palette='flare')       
            plt.savefig(self.config['output_dir'] + 'plots/alff_rois_' + self.config['tag_results'] + '.pdf')
    
        return alffs

    # Utilities
    def _apply_filter(self, data_root, b, a):
        '''Apply bandpass (if file doesn't exist or if we want to overwrite it)

        Inputs
        ----------
        data_root : str
            Path to the image to filter (i.e., no suffix, no file extension)

        Outputs
        ----------
        data_root_bp.nii.gz
            Filtered image

        '''
        if not os.path.isfile(data_root + '_bp.nii.gz') or self.config['overwrite_bp']:
            img = nib.load(data_root + '.nii.gz')
            data = img.get_fdata()
            filtered_data = filtfilt(b, a, data)
            data_bp_img = nib.Nifti1Image(filtered_data, img.affine, img.header)
            nib.save(data_bp_img, data_root + '_bp.nii.gz')

    def _apply_norm(self, data_root, interp):
        '''Apply normalization using existing warping field
        /!\ NN interpolation to avoid creating spurious correlations!

        Inputs
        ----------
        data_root : str
            Path to the image to filter (i.e., no suffix, no file extension)
        interp : str
            Type of interpolation (e.g.: 'nn', 'spline')

        Outputs
        ----------
        data_root_bp_pam50_nn.nii.gz
            Filtered image, normalized to the PAM50 template

        '''
        if not os.path.isfile(data_root + '_bp_pam50_' + interp + '.nii.gz') or self.config['overwrite_bp']:
            run_string = '/home/kinany/sct_5.6/bin/sct_apply_transfo -i ' + data_root + '_bp.nii.gz -d ' + \
                self.config['template_path'] + 'template/PAM50_t2.nii.gz -w ' + data_root.rsplit('/', 1)[0] + '/' + self.config['norm_dir'] + \
                '/warp_fmri2template.nii.gz -x ' +  interp + ' -o ' + data_root + '_bp_pam50_' + interp + '.nii.gz'
            os.system(run_string)

    def _flip(self, data_root):
        '''Apply normalization using existing warping field

        Inputs
        ----------
        data_root : str
            Path to the image to flip (i.e., no suffix, no file extension)

        Outputs
        ----------
        data_root_bp_pam50_nn_flipped.nii.gz
            Filtered image, normalized to the PAM50 template, flipped

        '''
        if not os.path.isfile(data_root + '_bp_pam50_nn_flipped.nii.gz') or self.config['overwrite_bp']:
            img_toflip = nib.load(data_root + '_bp_pam50_nn.nii.gz')
            data_toflip = img_toflip.get_fdata()
            data_flipped = data_toflip[::-1]
            img_flipped = nib.Nifti1Image(data_flipped, img_toflip.affine, img_toflip.header)
            nib.save(img_flipped,data_root + '_bp_pam50_nn_flipped.nii.gz')

    def _smooth(self, data_root, take_flipped):
        '''Smooth images using a 2x2x6 FWHM kernel
        
        Inputs
        ---------
        data_root : str
            Path to the image to flip (i.e., no suffix, no file extension)
        take_flipped : boolean
            False if we take images that are not flipped (i.e., for patients with right lesion or for healthy subjects)
            True if we take images that are flipped (i.e., for patients with left lesion)

        Outputs
        --------
        data_root_bp_pam50_nn(_flipped).nii.gz
            Filtered image, normalized to the PAM50 template, smoothed

        '''
        if (not os.path.isfile(data_root + '_bp_pam50_nn_smooth.nii.gz') and not os.path.isfile(data_root + '_bp_pam50_nn_flipped_smooth.nii.gz')) or self.config['overwrite_smooth']:
            if take_flipped == False:
                data_to_smooth = data_root + '_bp_pam50_nn'
            elif take_flipped == True:
                data_to_smooth = data_root + '_bp_pam50_nn_flipped'
            else:
                raise Exception(f'Value of take_flipped is {take_flipped} but should be True or False.')       

            # Check that input file exist
            if os.path.isfile(data_to_smooth + '.nii.gz'):
                tmp_smooth = image.smooth_img(data_to_smooth + '.nii.gz', [2,2,6])
                tmp_smooth.to_filename(data_to_smooth + '_smooth.nii.gz')
            else:
                raise Exception(f'Input file {data_to_smooth}.nii.gz does not exist.')  
        

    def _alff(self,data_root,take_flipped):
        '''Compute alff map for one subject

        Inputs
        ----------
        data_root : str
            Path to the image to flip (i.e., no suffix, no file extension)
        take_flipped : boolean
            False if we take images that are not flipped (i.e., for patients with right lesion or for healthy subjects)
            True if we take images that are flipped (i.e., for patients with left lesion)

        Outputs
        ----------
        data_root_alff_pam50.nii.gz
            ALFF image (i.e., std)
        data_root_alff_Z_pam50.nii.gz
            Z-scored ALFF image
        '''
        if (not os.path.isfile(data_root + '_alff_pam50.nii.gz')) or (not os.path.isfile(data_root + '_alff_Z_pam50.nii.gz')) or self.config['overwrite_alff_maps']: # If not already done or if we want to overwrite
            # Load data (different depending on whether we take flipped images or not)
            if take_flipped == False:
                # Check that input file exist
                if os.path.isfile(data_root + '_bp_pam50_nn_smooth.nii.gz'):
                    # Load the data
                    img = nib.load(data_root + '_bp_pam50_nn_smooth.nii.gz')
                    data = img.get_fdata()
                else:
                    raise Exception(f'Input file {data_root}_bp_pam50_nn_smooth.nii.gz does not exist.')   
            elif take_flipped == True:
                # Check that input file exist
                if os.path.isfile(data_root + '_bp_pam50_nn_flipped_smooth.nii.gz'):
                    # Load the data
                    img = nib.load(data_root + '_bp_pam50_nn_flipped_smooth.nii.gz')
                    data = img.get_fdata()
                else:
                    raise Exception(f'Input file {data_root} _bp_pam50_nn_flipped_smooth.nii.gz does not exist.')
            else:
                raise Exception(f'Value of take_flipped is {take_flipped} but should be True or False.')       

            # Compute ALFF
            # Define frequency range of interest
            lowcut = self.config['alff_freq_range'][0]
            highcut = self.config['alff_freq_range'][1]
            freqs = np.fft.fftfreq(data.shape[-1], d=self.config['TR'])
            freq_mask = np.logical_and(freqs >= lowcut, freqs <= highcut)

            # Compute the power spectrum of the signal
            power_spectrum = np.abs(np.fft.fft(data, axis=-1))

            # Select the low-frequency range of interest
            alff = np.mean(power_spectrum[..., freq_mask], axis=-1)

            # Save the ALFF image
            img_alff = nib.Nifti1Image(alff, img.affine, img.header)
            nib.save(img_alff, data_root + '_alff_pam50.nii.gz')

            # Save Z-scored ALFF image
            if os.path.isfile(data_root + '_alff_pam50.nii.gz'): # If previous step has worked (i.e., ALFF computed)        
                # Then: Z-score
                # Compute std and mean in mask
                stats_mean = ImageStats(in_file=data_root + '_alff_pam50.nii.gz', mask_file=self.config['template_cord'],op_string= '-m')
                mean = stats_mean.run().outputs.out_stat
                stats_std = ImageStats(in_file=data_root + '_alff_pam50.nii.gz', mask_file=self.config['template_cord'],op_string= '-s')
                std = stats_std.run().outputs.out_stat
                # Compute z-score ALFF
                run_string = 'fslmaths ' + data_root + '_alff_pam50.nii.gz -sub ' + str(mean) + ' -div ' + str(std) + ' ' + data_root + '_alff_Z_pam50.nii.gz'                    
                os.system(run_string)
            else:
                raise Exception(f'ALFF map {data_root}_alff_pam50.nii.gz cannot be found, Z-score cannot be computed.')       
            
