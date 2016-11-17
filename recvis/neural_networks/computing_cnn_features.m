clear;

net = load('imagenet-vgg-f.mat');

path = '../practical-category-recognition-2015a/data/images/';
files = dir(strcat(path,'*.jpg'));


for file = files'
    
    im  = imread(strcat(path,file.name));
    % Preprocessing
    im_ = single(im);
    im_ = imresize(im_, net.meta.normalization.imageSize(1:2));
    im_ = im_ - net.meta.normalization.averageImage;

    % Evaluation of nn
    res = vl_simplenn(net, im_);
    
    % Saving data
    soft = res(22).x;
    fc8 = res(21).x;
    fc7 = res(19).x;
    
    name = strsplit(file.name,'.');
    name = char(name(1));
    save(strcat('cnn/soft_',name) ,'soft');
    save(strcat('cnn/fc8_',name) ,'fc8');
    save(strcat('cnn/fc7_',name) ,'fc7');
    
end
