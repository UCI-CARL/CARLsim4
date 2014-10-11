classdef Utilities
    % A class containing some static helper methods.
    %
    % Version 10/5/2014
    % Author: Michael Beyeler <mbeyeler@uci.edu>
    
    %% CONSTANT PROPERTIES
    properties (Constant)
    end
    
    %% STATIC METHODS
    methods (Static)
        function [orLogic,dispOrLogic] = verify(var, orAndLogic)
            % [orLogic,dispOrLogic] = verify(var, orAndLogic) applies a
            % two-level Boolean logic to the properties of a variable var.
            %
            % The function returns the evaluation of the Boolean logic in
            % variable orLogic (true if it satisfies the logical
            % constraints), and a human-readable string of the parsed
            % logic constraints. The latter can be helpful in verifying
            % that the logic has been correctly specified and parsed.
            %
            % The Boolean logic is specified in a 2D cell array, where the
            % first dimension specifies all OR components, and the second
            % dimension specifies all AND components.
            % For example: orAndLogic = { {'isscalar',[0 10]}, 'isempty' }
            % translates to: "isscalar*[0 10] + isempty", which means var
            % must either be a scalar in the range [0,10] or empty.
            %
            % VAR        - The variable whose properties need to be
            %              verified.
            %
            % ORANDLOGIC - A 2D cell array containing a Boolean logic
            %              expression, represented by a keyword that
            %              indicates which property to check.
            %              String keywords: 'ischar', 'iscell', 'isempty',
            %              'islogical', 'isnumeric', 'isscalar', 'isvector'
            %              Numeric keywords: 2-element vector [min,max]
            %              indicates a range of values in which var has to
            %              lie.
            orLogic = false;  % evaluate logical expression
            dispOrLogic = ''; % display human-readable logical expression
            
            % if logic has only one component, make it iterable
            if ~iscell(orAndLogic)
                orAndLogic = {orAndLogic};
            end
            
            % loop over all first-level (OR) components
            % each component can have a list of subcomponents (AND)
            for i=1:numel(orAndLogic)
                orLvlComp = orAndLogic{i};
                
                % if sublogic has only one component, make it iterable
                if ~iscell(orLvlComp)
                    orLvlComp = {orLvlComp};
                end
                
                andLogic = true;
                dispAndLogic = '';
                
                % loop over all AND components
                % these have to be either a recognized string that triggers
                % a function call or a 2-element vector for range checking
                for j=1:numel(orLvlComp)
                    cmp = orLvlComp{j};
                    if ~isempty(dispAndLogic)
                        dispAndLogic=[dispAndLogic '*'];
                    end
                    
                    if ischar(cmp)
                        % a string indicating the property to check
                        switch lower(cmp)
                            case 'ischar'
                                andLogic = andLogic && ischar(var);
                            case 'iscell'
                                andLogic = andLogic && iscell(var);
                            case 'isempty'
                                andLogic = andLogic && isempty(var);
                            case 'islogical'
                                andLogic = andLogic && islogical(var);
                            case 'isnumeric'
                                andLogic = andLogic && isnumeric(var);
                            case 'isscalar'
                                andLogic = andLogic && isscalar(var);
                            case 'isvector'
                                andLogic = andLogic && isvector(var);
                            otherwise
                                error(['Unknown attribute type "' cmp '"'])
                        end
                        dispAndLogic = [dispAndLogic orLvlComp{j}];
                    elseif isnumeric(cmp) && numel(cmp)==2
                        % a 2-element vector indicating an allowed range
                        andLogic = andLogic && sum(var<cmp(1)|var>cmp(2))==0;
                        dispAndLogic = [dispAndLogic '[' num2str(cmp) ']'];
                    else
                        error(['Unknown attribute type "' num2str(cmp) '"'])
                    end
                end
                
                % add the andLogic as an OR term to the orLogic
                orLogic = orLogic || andLogic;
                if ~isempty(dispOrLogic),dispOrLogic=[dispOrLogic ' + '];end
                dispOrLogic = [dispOrLogic dispAndLogic];
            end
        end
    end
end
