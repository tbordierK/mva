clear;
category = 'motorbike' ;
%category = 'aeroplane' ;
%category = 'person' ;

previous_encoding = 'fv';
encoding = char('fc7');

pos = load(['../practical-category-recognition-2015a/data/' category '_train_' previous_encoding '.mat']) ;
pos = rmfield(pos,'histograms');

n_names = length(pos.names);
%features = zeros(n_names,1000);
features = zeros(n_names,4096);

for i=1:n_names
    elm = char(pos.names(i));
    feat = load(['cnn/' encoding '_' elm]);
    
    %features(i,:) = feat.soft(1,1,:); 
    %features(i,:) = feat.fc8(1,1,:); 
    features(i,:) = feat.fc7(1,1,:); 
end

pos = setfield(pos,encoding,features);

save(['../practical-category-recognition-2015a/data/' category '_train_' encoding '.mat'],'pos');
%% Building the validation data set
clear;

%category = 'motorbike' ;
%category = 'aeroplane' ;
category = 'person' ;

previous_encoding = 'fv';
encoding = char('soft');

pos = load(['../practical-category-recognition-2015a/data/' category '_val_' previous_encoding '.mat']) ;
pos = rmfield(pos,'histograms');

n_names = length(pos.names);
features = zeros(n_names,1000);
%features = zeros(n_names,4096);

for i=1:n_names
    elm = char(pos.names(i));
    feat = load(['cnn/' encoding '_' elm]);
    
    features(i,:) = feat.soft(1,1,:); 
    %features(i,:) = feat.fc8(1,1,:); 
    %features(i,:) = feat.fc7(1,1,:); 
end

pos = setfield(pos,encoding,features);

save(['../practical-category-recognition-2015a/data/' category '_val_' encoding '.mat'],'pos');



%% Building background training set

previous_encoding = 'fv';
encoding = char('soft');

pos = load('../practical-category-recognition-2015a/data/background_train_fv.mat') ;
pos = rmfield(pos,'histograms');

n_names = length(pos.names);

features = zeros(n_names,1000);
% for fc7
%features = zeros(n_names,4096);

for i=1:n_names
    elm = char(pos.names(i));
    feat = load(['cnn/' encoding '_' elm]);
    
    features(i,:) = feat.soft(1,1,:); 
    %features(i,:) = feat.fc8(1,1,:); 
    %features(i,:) = feat.fc7(1,1,:); 
end

pos = setfield(pos,encoding,features);
save(['../practical-category-recognition-2015a/data/background_train_' encoding '.mat'],'pos');

%% Building validation set


previous_encoding = 'fv';
encoding = char('fc7');

pos = load('../practical-category-recognition-2015a/data/background_val_fv.mat') ;
pos = rmfield(pos,'histograms');

n_names = length(pos.names);

%features = zeros(n_names,1000);
% for fc7
features = zeros(n_names,4096);

for i=1:n_names
    elm = char(pos.names(i));
    feat = load(['cnn/' encoding '_' elm]);
    
    %features(i,:) = feat.soft(1,1,:); 
    %features(i,:) = feat.fc8(1,1,:); 
    features(i,:) = feat.fc7(1,1,:); 
end

pos = setfield(pos,encoding,features);
save(['../practical-category-recognition-2015a/data/background_val_' encoding '.mat'],'pos');