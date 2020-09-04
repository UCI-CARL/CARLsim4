xlName = "ca3net_01_22_20_hc.xlsm";
connTypeSTPmod = readtable('PrePostSTPCA3modified_v4.csv');
connTypeSTPmod = table2cell(connTypeSTPmod);

[~,~,synCondMat] = xlsread(xlName,'syn_conds_int_to_int');
synCondMat = synCondMat(1:11,:);

[~,~,UMat] = xlsread(xlName,'syn_resource_release_int_to_int');
UMat = UMat(1:11,:);

[~,~,taurecMat] = xlsread(xlName,'syn_rec_const_int_to_int');
taurecMat = taurecMat(1:11,:);

[~,~,tauIMat] = xlsread(xlName,'syn_decay_const_int_to_int');
tauIMat = tauIMat(1:11,:);

[~,~,taufacilMat] = xlsread(xlName,'syn_facil_const_int_to_int');
taufacilMat = taufacilMat(1:11,:);

[~,~,connMat] = xlsread(xlName,'internal_to_internal_conn_prob');

connMat = connMat(71:81,1:11);
preconnMat = connMat(2:end,1);
postconnMat = connMat(1,2:end);
postconnMat = postconnMat';

for i = 1:length(preconnMat)
    preconnMat{i} = regexprep(preconnMat{i}, '-','_');
    preconnMat{i} = regexprep(preconnMat{i}, ' ','_');
    preconnMat{i} = regexprep(preconnMat{i}, '+','');
    postconnMat{i} = regexprep(postconnMat{i}, '-','_');
    postconnMat{i} = regexprep(postconnMat{i}, ' ','_');
    postconnMat{i} = regexprep(postconnMat{i}, '+','');
end

% Modify the cell names from keivan's document so that they are compatible
% with those in the ca3 net XL
for i = 1:length(connTypeSTPmod(:,1))
preName = strsplit(connTypeSTPmod{i,1});
preName = preName(1:end-1);
preName = strjoin(preName,'_');
preName = regexprep(preName, '-','_');
preName = regexprep(preName, ' ','_');
preName = regexprep(preName, '+','');
connTypeSTPmod{i,1} = preName;

postName = strsplit(connTypeSTPmod{i,2});
postName = postName(1:end-1);
postName = strjoin(postName,'_');
postName = regexprep(postName, '-','_');
postName = regexprep(postName, ' ','_');
postName = regexprep(postName, '+','');
connTypeSTPmod{i,2} = postName;
end

connTypeSTPmod(:,1:2) = strrep(connTypeSTPmod(:,1:2), ...
                               'CA3_Basket_CCK', ...
                               'CA3_BC_CCK');
connTypeSTPmod(:,1:2) = strrep(connTypeSTPmod(:,1:2), ...
                               'CA3_Mossy_Fiber_Associated_ORDEN', ...
                               'CA3_MFA_ORDEN');


synCondMat2 = synCondMat;
UMat2 = UMat;
taurecMat2 = taurecMat;
tauIMat2 = tauIMat;
taufacilMat2 = taufacilMat;

k = 1;
for i = 1:size(synCondMat2,1)-1
    for j = 2:2:size(synCondMat2,2)
        if (strcmp(preconnMat{i},connTypeSTPmod{k,1}) == 1) && ...
           (strcmp(postconnMat{j/2},connTypeSTPmod{k,2}) == 1)
            synCondMat2{i+1,j} = connTypeSTPmod{k,3};
            UMat2{i+1,j} = connTypeSTPmod{k,7};
            taurecMat2{i+1,j} = connTypeSTPmod{k,5};
            tauIMat2{i+1,j} = connTypeSTPmod{k,4};
            taufacilMat2{i+1,j} = connTypeSTPmod{k,6};
            k = k + 1;
        end
        
%         k = k + 1;
    end
    
end


writecell(synCondMat2,xlName,'Sheet','syn_conds_int_to_int')
writecell(UMat2,xlName,'Sheet','syn_resource_release_int_to_int')
writecell(taurecMat2,xlName,'Sheet','syn_rec_const_int_to_int')
writecell(tauIMat2,xlName,'Sheet','syn_decay_const_int_to_int')
writecell(taufacilMat2,xlName,'Sheet','syn_facil_const_int_to_int')