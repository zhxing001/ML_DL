% ����ѵ���Ͳ��Ա�ǩ�����ݡ�
% 1--zhxing
% 2--liuyang
% 3--juchunwu
% 4--hailong
% 5--lyzhen
% 6--others
% 
% ���е�ͼƬ����һ�У��������֮�У���������һ������ÿ�ж�Ӧ�ı�ǩ����ǩ�Ķ�Ӧ��ϵ�������ע������˵����
% ����������е�ͼƬ����64*64*3�Ĵ�С��


% train_img_and_label=[];
% test_img_and_label=[];
% 
% %������������ʾ�������
% h=waitbar(0,'processing img...');
% 
% for i=1:6 
% img_path='C:\Users\zhxing\Desktop\data';
% img_path_i=[img_path '\' num2str(i) '\'];
% img_dir=dir([img_path_i '*.jpg']);          %��ȡ�����ļ����µ�����ͼ��
% num_img=length(img_dir);
% display(['���ڴ���� ' num2str(i) ' �����ݣ�  ��' num2str(num_img) '��ͼƬ' ])
% tic;
% for j=1:num_img
%     img=imread([img_path_i img_dir(j).name]);
%     if mod(j,5)==0   %ÿ5�Ű�һ���ӽ����Լ�
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
% display('finished����')
% 
%�����ݴ��ң�����ѵ����ʱ��һ��batch����Ķ���ͬһ���������ݡ�



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

%�����ݺͱ�ǩ�ֿ�
train_img=train(:,1:end-1);       
train_label=train(:,end);
test_img=test(:,1:end-1);
test_label=test(:,end);

%���������ó���ά���󲢱���Ϊmat��ʽ��Ȼ�����python���д���
train_sz=size(train_img);
train_img_rz=reshape(train_img,[train_sz(1),64,64,3]);
save('C:\Users\zhxing\Desktop\data\train_img1.mat','train_img_rz');


test_sz=size(test_img);
test_img_rz=reshape(test_img,[test_sz(1),64,64,3]);
save('C:\Users\zhxing\Desktop\data\test_img1.mat','test_img_rz');

%�����ֱ�ǩת��Ϊ�����ʽ�ġ�
train_label_sz=size(train_label);          %��С
train_labels=zeros(train_label_sz(1),6);   %������label��
for i=1:train_label_sz(1)
    train_labels(i,train_label(i))=1;
end
save('C:\Users\zhxing\Desktop\data\train_label1.mat','train_labels');
test_label_sz=size(test_label);          %��С
test_labels=zeros(test_label_sz(1),6);   %������label��
for i=1:test_label_sz(1)
    test_labels(i,test_label(i))=1;
end
save('C:\Users\zhxing\Desktop\data\test_label1.mat','test_labels');