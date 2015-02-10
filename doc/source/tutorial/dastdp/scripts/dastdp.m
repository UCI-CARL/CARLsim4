% Version 8/13/14

addpath('../results');
addpath('../../../tools/offline_analysis_toolbox');

CR = ConnectionMonitor('input','excit');
CR.setPlotType('histogram');
CR.plot;