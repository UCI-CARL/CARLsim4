currentpath = pwd;
if ~strcmpi(currentpath(end-6:end), 'scripts')
    error('Make sure to run the OAT from the scripts directory.')
end

% add path to Offline Analysis Toolbox
addpath('../../../../../tools/offline_analysis_toolbox/')

% add path to Visual Stimulus Toolbox
addpath('../../../../../external/VisualStimulusToolbox/VisualStimulus')