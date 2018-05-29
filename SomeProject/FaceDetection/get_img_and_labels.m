% 制作训练和测试标签及数据。
% 1--zhxing
% 2--liuyang
% 3--juchunwu
% 4--hailong
% 5--lyzhen
% 6--others
% 
% 所有的图片拉成一行，放入矩阵之中，矩阵的最后一列是其每行对应的标签，标签的对应关系在上面的注释中已说明。
% 这个程序所有的图片均是64*64*3的大小。


% train_img_and_label=[];
% test_img_and_label=[];
% 
% %创建进度条显示处理进程
% h=waitbar(0,'processing img...');
% 
% for i=1:6 
% img_path='C:\Users\zhxing\Desktop\data';
% img_path_i=[img_path '\' num2str(i) '\'];
% img_dir=dir([img_path_i '*.jpg']);          %获取所有文件夹下的所有图像
% num_img=length(img_dir);
% display(['正在处理第 ' num2str(i) ' 类数据：  共' num2str(num_img) '张图片' ])
% tic;
% for j=1:num_img
%     img=imread([img_path_i img_dir(j).name]);
%     if mod(j,5)==0   %每5张把一个扔进测试集
%         img_reshape=reshape(img,[1,4096*3]);
%         test_img_and_label_tmp=[img_reshape i];
%         test_img_and_label=[test_img_and_label;test_img_and_label_tmp];
%     else
%         img_reshape=reshape(img,[1,4096*3]);
%         train_img_and_label_tmp=[img_reshape i];
%         train_img_and_label=[train_img_and_label;train_img_and_label_tmp];
%     end
%     waitbar(j/num_img);
% end
% toc;
% end
% display('finished！！')
% 
%把数据打乱，避免训练的时候一个batch传入的都是同一个类别的数据。



train_img_and_label=load('C:\Users\zhxing\Desktop\data\train.mat');
test_img_and_label=load('C:\Users\zhxing\Desktop\data\test.mat');
train_img_and_label=train_img_and_label.train_img_and_label;
test_img_and_label=test_img_and_label.test_img_and_label;

rowrank_test = randperm(size(test_img_and_label, 1));
test_img_and_label=test_img_and_label(rowrank_test,:);

rowrank_train = randperm(size(train_img_and_label, 1));
train_img_and_label=train_img_and_label(rowrank_train,:);


test=test_img_and_label;
train=train_img_and_label;

%把数据和标签分开
train_img=train(:,1:end-1);       
train_label=train(:,end);
test_img=test(:,1:end-1);
test_label=test(:,end);

%把数据重置成四维矩阵并保存为mat形式，然后读入python进行处理
train_sz=size(train_img);
train_img_rz=reshape(train_img,[train_sz(1),64,64,3]);
save('C:\Users\zhxing\Desktop\data\train_img1.mat','train_img_rz');


test_sz=size(test_img);
test_img_rz=reshape(test_img,[test_sz(1),64,64,3]);
save('C:\Users\zhxing\Desktop\data\test_img1.mat','test_img_rz');

%把数字标签转换为码表形式的。
train_label_sz=size(train_label);          %大小
train_labels=zeros(train_label_sz(1),6);   %建立空label表
for i=1:train_label_sz(1)
    train_labels(i,train_label(i))=1;
end
save('C:\Users\zhxing\Desktop\data\train_label1.mat','train_labels');
test_label_sz=size(test_label);          %大小
test_labels=zeros(test_label_sz(1),6);   %建立空label表
for i=1:test_label_sz(1)
    test_labels(i,test_label(i))=1;
end
save('C:\Users\zhxing\Desktop\data\test_label1.mat','test_labels');