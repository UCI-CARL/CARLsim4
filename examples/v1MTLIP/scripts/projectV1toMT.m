function projectV1toMT()
% projectV1toMT()
%
% This function is computing the projections from V1 to MT. V1 has (at each
% pixel location) 28 different space-time oriented filters. Those need to
% be projected onto MT neurons selective to 8 directions and 3 speeds.
% All 28 filters that were used are described by v1Dirs (and v1DirsSphere).
% A MT neuron is described by its direction and speed preference.
% v1DirsSphere is then steered into these directions by using shQwts() (an
% original function from the S&H model that computes the weights for
% interpolating squared directional derivative filters).
%
% This function uses scripts from an open-source MATLAB package of
% Simoncelli & Heeger's Motion Energy model, obtained from
% http://www.cns.nyu.edu/~lcv/MTmodel/ version 1.0 (10/14/05).
% Authors: Timothy Saint (saint@cns.nyu.edu) and Eero P. Simoncelli
% (eero.simoncelli@nyu.edu)
%
% Author: Michael Beyeler <mbeyeler@uci.edu>
% Ver 07/13/2013


% 3 populations selective to different speeds (pixels/frame)
speeds=[1.5 0.125 9];

% the V1 population of filter orientations (spherical coordinates)
% this is equal to pars.v1PopulationDirections in the original S&H model
v1DirsSphere = [5.5475    0.2162
    3.2461    0.2176
    0.6942    0.2266
    4.0698    0.2326
    1.6542    0.2393
    6.0430    0.2428
    5.0491    0.2528
    2.6917    0.2614
    4.5468    0.2656
    2.1323    0.2685
    0.2522    0.2885
    1.1696    0.3033
    3.6457    0.4343
    3.0694    0.7118
    5.5522    0.7333
    2.3645    0.7374
    1.5678    0.7681
    0.7111    0.7819
    4.8972    0.7988
    6.1972    0.8059
    4.1718    0.8077
    3.6350    1.4046
    1.1300    1.4926
    1.9858    1.5147
    2.8434    1.5681
    0.2510    1.5743
    4.5454    1.6289
    5.5369    1.6311];

% In order to reduce the number of connections, cut off unreasonably small
% weights. The choice of a threshold is somewhat arbitrary. We choose a
% fraction of 0.1, arguing that smaller values probably do not influence
% the overall response too much.
wtCutOff = 0.2;


for r=1:length(speeds)
    % mtNeurons respond to 1 of 8 directions, at 1 of 3 speeds
    % mtNeurons must be in S&H "standard" format, [direction speed]
    
    % The 8 subpools of MT neurons will respond to: [R, UR, U, UL, L, DL,
    % D, DR]; R=right, L=left, U=up, D=down
    % However, this is not [0 pi/4 pi/2 ..etc] because Matlab plots [Y,X]
    % instead [X,Y], so rotate it:
    mtNeurons = [mod((pi:-pi/4:-3/4*pi)+2*pi,2*pi) ; ones(1,8)*speeds(r)]';
    
    wts=[];
    for i=1:8
        % use v1Dirs in spherical coordinates
        tmp = pinv(shQwts(v1DirsSphere))'*shQwts(mtNeurons(i,:))';
        
        % subtract mean response of population
        tmp = tmp-mean(tmp);
        
        % cut off unreasonably small weights (less than a fraction wtCutOff
        % of max. weight)
        tmp = tmp.*((tmp>0).*(tmp>wtCutOff*max(tmp)) ... % for pos. wt
            + (tmp<0).*(tmp<wtCutOff*min(tmp)));         % for neg. wt

        % add to wt vector
        wts=[wts tmp];
    end
    
    % print for use in examples/v1MTLIP/main_v1MTLIP.cpp
    fprintf(1,'float motionProj%d[28][8] = {',r);
    fprintf(1,'{%f, %f, %f, %f, %f, %f, %f, %f},\n',wts'); % ' to write in right order
    fprintf(1,'};\n');
end


%% S&H DIRECTION INTERPOLATION
    % qWts = shqWts(dirs)      get the weights for interpolating squared
    % directional derivative filters
    % dirs is an mx3 matrix specifying the direction of m neurons in 3d
    % fourier space; qWts is a matrix whose rows are the weights for
    % interpolating from a population of qWtsponses that you already have.
    %
    % Example use: you have Q, N, or C. Assume for this example that you have
    % Q. You also know the direction of the filters in Q, which we'll call
    % v1dirs. You have dirs, which contains the directions of the
    % neurons whose qWtsponses you want to get through interpolation. Let's call
    % R the qWtsponses of these neurons.
    % R = shqWts(dirs) * shqWts(v1dirs)^-1 * Q;
    % That ought to do the trick.
    function qWts = shQwts(dirs)
        
        dirs(:,2) = atan3(dirs(:,2), ones(size(dirs, 1), 1));
        dirs = sphere2rec(dirs);
        
        d = sqrt(sum(dirs.^2, 2));
        d = repmat(d, 1, size(dirs, 2));
        dirs = dirs./d;
        
        fac = zeros(1, 7);
        fac(1:2) = 1;
        for n = 2:6
            fac(n+1) = n*fac(n);
        end
        
        % generate the weighting vectors
        qWts = zeros(size(dirs,1), 28);
        pt = 0;
        for o3 = 0:6
            for o2 = 0:6-o3
                o1 = 6-o3-o2;
                pt = pt+1;
                const = fac(7)./(fac(o3+1)*fac(o2+1)*fac(o1+1));
                qWts(:,pt) = const .* dirs(:,1).^o1 .* dirs(:,2).^o2 .* dirs(:,3).^o3;
            end
        end
        
    end

    % theta = atan3(y, x)      Get the four quadrant atan with 0 <= theta <= 2.*pi
    %
    % atan3 is slightly different than atan2 in that the output ranges from 0
    % to 2*pi instead of from -pi to pi.
    %
    % SEE ALSO: ATAN, ATAN2
    function theta = atan3(y, x)
        
        theta = atan2(y,x);
        theta(theta<0) = theta(theta<0) + 2*pi;
    end

    % rectangularPoints = sphere2rec(spherPts)
    %
    % Transform [az, el, radius] coordinates to [y, x, t]
    % Required arguments:
    % spherPts       The points in spherical coordinates you want
    %                       transformed into rectangular coordinates. Each row
    %                       contains a different point. The first column
    %                       specifies the azimuthal angle in radians with
    %                       0 = right. The second column specifies
    %                       the elevation angle in radians from -pi/2 to pi/2,
    %                       with 0 lying on the XY plane. The third column
    %                       specifies the radius.
    %
    % Output:
    % rectangularPoints     the transformed points Each row contains a different
    %                       point in [Y X T] coordinates.
    function rectangularPoints = sphere2rec(spherPts)
        
        if size(spherPts,2) == 2;
            spherPts = [spherPts, ones(size(spherPts,1), 1)];
        end
        
        rectangularPoints = zeros(size(spherPts));
        
        rectangularPoints(:,1) = spherPts(:,3).*cos(spherPts(:,2)).*sin(spherPts(:,1));
        rectangularPoints(:,2) = spherPts(:,3).*cos(spherPts(:,2)).*cos(spherPts(:,1));
        rectangularPoints(:,3) = spherPts(:,3).*sin(spherPts(:,2));
    end

end