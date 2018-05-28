clear;
clc;
imgPath = '/home/user6/DATA/Eye_fixation_data/SALICON/salicon-api-master/PythonAPI/SALICON_crop_stimulus/'; 
imgDir  = dir([imgPath '*.jpg']);
n_img=length(imgDir);
for i =1:n_img
    img =imread([imgPath imgDir(i).name]);
    img1=imresize(img,[224,224]);
    img2=double(img1)/255;
    img_t=permute(img2,[3,2,1]);
    num=numel(img_t);
    img_flat=reshape(img_t,[1,num]);
    SALICON_data(i,:)=img_flat;
end
save('SALICON_data.mat','SALICON_data'); 