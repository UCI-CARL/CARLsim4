% creating a grating video

%% INITIALIZATION

initOAT


%% CREATING THE GRATING

grating = GratingStim([32 32]);

for dir=(0:7)*45
    grating.add(10, dir)
end

grating.plot



%% SAVING TO FILE

grating.save('../input/grating.dat')
