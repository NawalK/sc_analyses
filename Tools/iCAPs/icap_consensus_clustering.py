from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ConsensusClustering:
    '''
    The ConsensusClustering class is used to compute and visualize
    metrics to support the choice of K
    Note :  This class relies on the output of the iCAP's consensus clustering pipeline

    Attributes
    ----------
    config : dict
        Contains information regarding subjects, sessions, rois, etc.
    avg_consensus: df
        Contains the average consensus (i.e., across cluster) for each K         
    cdf: df
        Contains the cumulative density function derived from the consensus matrices for each K 
    '''

    def __init__(self, config):
        self.config = config
        self.avg_consensus = None
        self.cdf = None
        self.parse_avg_consensus()
        self.parse_cdf()

    def parse_avg_consensus(self):
        '''Parse average consensus for each K into a dataframe'''
        consval = np.empty((0,), float)
        kval = np.empty((0,), int)
        for k in self.config['k_range']:
            consval = np.append(consval,loadmat(self.config['icap_root']+self.config['icap_folder'] + 'K_' + str(k) + '_Dist_cosine_Folds_20/iCAPs_consensus.mat')['iCAPs_consensus'])
            kval = np.append(kval,(np.full((k,),k)))
        colnames = ["k","consensus"]
        self.avg_consensus = pd.DataFrame(list(zip(kval,consval)),columns=colnames)

    def parse_cdf(self):
        '''Parse CDF for each K into a dataframe'''
        CDF = loadmat(self.config['icap_root']+self.config['icap_folder'] + self.config['cons_folder'] + '/CDF.mat')['CDF']
        kval = np.empty((0,), int) # K range
        consval = np.empty((0,), int) # From 0 to 100 for each K
        cdfval = np.empty((0,), float) # Fraction < consensus value
        for k_ix, k in enumerate(self.config['k_range']):
            for cons_ix in range(0,len(CDF[k_ix,])):
                kval = np.append(kval,k)
                consval = np.append(consval,cons_ix)
                cdfval = np.append(cdfval,CDF[k_ix,cons_ix])
        colnames = ["k","consensus","fraction"]
        self.cdf = pd.DataFrame(list(zip(kval,consval,cdfval)),columns=colnames)
    
    def plot_avg_consensus(self, save_results = False):
        '''Plot the average consensus +- SD for each K
         Parameters
        ----------
        save_results : boolean
            Set to True to save figure (default = False)'''

        plt.figure(figsize = (7,4))
        sns.pointplot(data = self.avg_consensus, x="k", y="consensus", palette=sns.color_palette('flare', n_colors=len(self.config['k_range'])), ci='sd', err_style='bars');
        
        # If option is set, save results as a png
        if save_results == True:
            plt.savefig(self.config['output_dir'] + 'average_consensus_' + self.config['tag_results'] + '.png')

    def plot_cdf(self, to_highlight=[],save_results = False):
        '''Plot the CDF +- SD for each K
        
        Parameters
        ----------
        to_highlight : list
            Contains K values for which line should be highlighted
        save_results : boolean
            Set to True to save figure (default = False)'''

        # Style palette
        dash_list = sns._core.unique_dashes(self.cdf['k'].unique().size+1)
        style = {key:value for key,value in zip(self.cdf['k'].unique(), dash_list[1:])}
        linesize = {key:value for key,value in zip(self.cdf['k'].unique(), np.full((len(self.config['k_range']),),1))}
        for i in to_highlight:
            if i in self.config['k_range']:
                style[i] = '' # Empty string means solid
                linesize[i] = 2 # Thicker lines for selected values
            else:
                print(f'{i} is not in K range ({self.config["k_range"][0]},{self.config["k_range"][-1]})')
        sns.set_style("ticks")
        plt.figure(figsize=(7,6))
        sns.lineplot(data=self.cdf,x='consensus',y='fraction',hue='k',style='k',size='k',sizes=linesize,dashes=style,palette=sns.color_palette('flare', n_colors=len(self.config['k_range'])));
        plt.legend(bbox_to_anchor=(1.01,1.02), loc='upper left',frameon=False);
        plt.xlim([0,100])
        plt.tight_layout()
    
        # If option is set, save results as a png
        if save_results == True:
            plt.savefig(self.config['output_dir'] + 'cdf_' + self.config['tag_results'] + '.png')
