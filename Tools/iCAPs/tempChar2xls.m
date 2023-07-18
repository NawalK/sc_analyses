%% tempChar2xls
% Function to format tempChar.mat into a xls file for later use in Python
% 
% ============ INPUTS =============
% pathname      Path to tempChar file
% filename      Name of tempChar file
% 
% output_tag    String containing a tag to name the excel file (optional)
%               
%               
% ============ OUTPUTS =============
% tempChar.csv  csv file containing the metrics of interest
%
%

function tempChar2xls(pathname,filename,output_tag)
    % Load tempChar.mat
    load([pathname filename])
    
    % Define output
    if ~exist('output_tag','var')
        output_name = [pathname 'tempChar.xlsx'];
    else 
        output_name = [pathname 'tempChar' output_tag '.xlsx'];
    end
    
    % Define metrics of interest (for those that just need to be copied)
    metrics_of_interest = {'duration_total_perc','duration_avg_counts'};

    % Create an excel file with a sheet for each metric of interest
    for i = 1:length(metrics_of_interest)
        writematrix(tempChar.(metrics_of_interest{i}),output_name,'Sheet',metrics_of_interest{i})
    end
    
    % For the coactivation, take the mean
    coactive_icaps = zeros(length(tempChar.coactiveiCAPs_total),1);
    for sub = 1:length(tempChar.coactiveiCAPs_total)
        coactive_icaps(sub) = mean(tempChar.coactiveiCAPs_total{sub});
    end
    
     writematrix(coactive_icaps,output_name,'Sheet','coactive_icaps')