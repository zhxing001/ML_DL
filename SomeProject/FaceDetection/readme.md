## 环境：

* 主要环境：win10+python3.6+tensorflow1.2  matlab2015b
* 其他依赖库：opencv_python,scipy,numpy等

##  文件：

* get_my_faces_opencv.py          获取训练图片，调用摄像头
* get_img_and_labels.m            制作训练测试数据和标签
* get_date.py                     获取制作的训练测试数据
* train_and_detection_faces.py    训练和检测，可以是视频源或者调用摄像头

## blog：

[face_detection](https://www.jianshu.com/p/2c9f9180a944)


#  功能和框架
想做的是这么一个东西：识别视频（或者摄像头获得的实时视频）中的人脸，并判断是谁（因为数据采集的原因，找了身边的5个朋友采集了一些数据），如果不是这几个人，标记为其他人。
功能上其实比较简单，主要是想体会一下这整个过程，做下来还是有很多值得注意的地方的。大致框架也比较简单：

![框架](https://upload-images.jianshu.io/upload_images/5252065-5b1462091807aae3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

下面就分别说一下。
#  1.视频采集
这次的整个编程语言选择的是python，之所以选择python主要是因为要遇到CNN来进行分类，所以还是在python下容易实现一些。linux还是用的不熟，所以还是用的windows10+pycharm。
视频采集用的是opencv的模块，应该说是很简单易用的模块。c++版本的videocapture原来写过，在[这里](https://www.jianshu.com/p/581108baa71e),python版本的大同小异。典型的简单使用：
```
#需要调用摄像头的话只需要参数设置为0就可以打开笔记本自带的摄像头
camera = cv2.VideoCapture('liuyang.mp4')
success, img = camera.read()
```
这样就没有问题了。

#  2. 人脸定位
人脸定位采用的harr特征来做的，使用的是opencv训练好的harr特征分类器，这个在opencv的源文件里有：`haarcascade_frontalface_default.xml`,这个文件保存的就是harr特征正脸检测的模型，是已经训练好的。
harr特征是基础图像处理领域三个基本特征之一(还有hog和Lbp)
加载特征使用的是`CascadeClassifier`这个分类器，是一个级联分类器。在官方文档中有定义。

>Use the [cv::CascadeClassifier](https://docs.opencv.org/3.1.0/d1/de5/classcv_1_1CascadeClassifier.html) class to detect objects in a video stream. Particularly, we will use the functions:
>*   [cv::CascadeClassifier::load](https://docs.opencv.org/3.1.0/d1/de5/classcv_1_1CascadeClassifier.html#a1a5884c8cc749422f9eb77c2471958bc) to load a .xml classifier file. It can be either a Haar or a LBP classifer
>*   [cv::CascadeClassifier::detectMultiScale](https://docs.opencv.org/3.1.0/d1/de5/classcv_1_1CascadeClassifier.html#aaf8181cb63968136476ec4204ffca498) to perform the detection.

定义很简单，这个是c++版本的解释，python载入模型的时候：
`haar = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')`

检测的时候一般使用的是：detectMultiScale这个函数。使用起来也比较简单，python的结果是返回在函数外边，c++版本是将`Vector<Rect>`作为参数传入。
```
CV_WRAP virtual void detectMultiScale( const Mat& image,  
                                   CV_OUT vector<Rect>& objects,  
                                   double scaleFactor=1.1,  
                                   int minNeighbors=3, int flags=0,  
                                   Size minSize=Size(),  
                                   Size maxSize=Size() ); 
const Mat& image: 需要被检测的图像（灰度图）
vector<Rect>& objects: 保存被检测出的人脸位置坐标序列
double scaleFactor: 每次图片缩放的比例
int minNeighbors: 每一个人脸至少要检测到多少次才算是真的人脸
int flags： 决定是缩放分类器来检测，还是缩放图像
Size(): 表示人脸的最大最小尺寸
```
python版本的使用就更加简单：
```
haar = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces = haar.detectMultiScale(gray_img, 1.3, 5)
#返回的face就是人脸的位置以及大小（左上角的位置及宽和高），相当于c++版本中的vector<Rect>
```
所以这样下来进行人脸定位就比较简单了。下面的这段程序不仅仅是实现了一个人脸的定位，这个实际上是我获取训练样本使用的一个程序，也是先从摄像头获取视频，然后定位人脸，把人脸图像取下来resize到64的尺寸，然后随机变换进行存储（增加样本多样性）。
```
import cv2
import os
import sys
import random
import time


#新建存储文件夹
out_dir = './my_faces'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


# 改变图片的对比度和亮度，增加图像多样性
def relight(img, alpha=1, bias=0):
    w = img.shape[1]
    h = img.shape[0]
    #image = []
    for i in range(0,w):
        for j in range(0,h):
            for c in range(3):
                tmp = int(img[j,i,c]*alpha + bias)
                if tmp > 255:
                    tmp = 255
                elif tmp < 0:
                    tmp = 0
                img[j,i,c] = tmp
    return img


#  获取分类器，用的是opencv的harr默认特征
haar = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#  打开摄像头 参数为输入流，可以为摄像头或视频文件
camera = cv2.VideoCapture('liuyang.mp4')

n = 1

while 1:
    if (n <= 10000):
        print('It`s processing %s image.' % n)
        # 读帧
        start=time.time()
        success, img = camera.read()
        cv2.imshow('face',img)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = haar.detectMultiScale(gray_img, 1.3, 5)       #多尺度检测人脸

        for f_x, f_y, f_w, f_h in faces:
            face = img[f_y:f_y+f_h, f_x:f_x+f_w]
            face = cv2.resize(face, (64,64))
            '''
            if n % 3 == 1:
                face = relight(face, 1, 50)
            elif n % 3 == 2:
                face = relight(face, 0.5, 0)
            '''
            face = relight(face, random.uniform(0.5, 1.5), random.randint(-50, 50))
            cv2.imshow('img', face)
            cv2.imwrite(out_dir+'/'+str(n)+'.jpg', face)
            #time_cost
            end=time.time()
            time_cost=end-start
            print('time_cost:\t'+str(time_cost))

            n+=1
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            break
    else:
        break
```
#  3. 人脸识别
人脸识别也是先定位再识别，没有想要做的太复杂，所以定位还是使用的上面说的利用harr来定位，识别用的是CNN，用CNN进行识别之后然后再返回结果到主程序进行显示，这其中最重要的两个部分是训练数据制作和模型训练。
##3.1训练数据制作。
我找了5个人采集数据，同一个人不同表情的人脸图片大概都是5000张左右，然后下载了一个数据集（[LFW](http://vis-www.cs.umass.edu/lfw/)）当做“其他类”，下载下来大概有13000多张人脸，而且是在不同的文件夹下的，所以全部读出来放在同一个文件夹下并删掉了一部分只剩下60000多张，与其他类的数量来匹配。所以一共是6类，分别放在6个文件夹里：

![](https://upload-images.jianshu.io/upload_images/5252065-bb5c5f6a1705255c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

然后用matlab读入图片数据，顺便打上标签，所有的结果存放在一个2维矩阵中，为了达到这个目的，每张图片读进来先拉长成为1行：64_64_3的图片，拉长成为长度12288的向量，然后在其最后一列打上其所属类的标签，标签就先用1,2,3,4,5,6这样的数字打上，所以每张图片和其标签就是一个行向量，维度为12289。在这个过程中，每5张照片，选择一张放入测试集，这样的话就相当于分掉1/5的数据去做训练集。下面是这么一个处理过程，完成之后我把这样的数据就存成mat形式了，因为这样的一次处理耗时还是挺可观的，大概需要半个小时的时间才处理完3万多张图片。

```
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
```
得到这样的一个图像矩阵之后，下面的任务就是把矩阵按照行打乱顺序，来防止训练的时候一个batch传入的都是同一类的数据，matlab提供了这么一个函数。先获得一个乱序表，然后按照这个乱序表把矩阵打乱。
`rowrank=randperm(num)`这个函数可以获得一个乱序表，然后利用`B = A(rowrank, :)`把矩阵打乱，前面的把图像拉成一行的原因也是为了在此同步打乱数据和标签。再打乱之后就可以把数据和标签分开了，以后的使用中，依据索引就可以完全对应了，一般在制作训练数据到这一步的时候，最好随机选择一些数据输出来看标签和数据是否对应。
```

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
```
接着，可以把数据恢复成原始尺寸来进行保存，和python要求的格式相同`[batch，width,height,channel]`。这里也是一个技巧，如果不这样保存的话读入python里然后再想恢复其实是不容易的，因为python中numpy通道的存储方式和matlab是不同的，matlab中一个通道存储完然后接着存储另一个通道，但是python里是一个位置的三个通道连续存储，所以如果直接把一行向量存入mat里再读入python直接用reshape是恢复不出来三通道的图像的。
另外，标签也转换成码表形式的。（这个是叫码表么？我不太确定）
```
%把数据和标签分开
train_img=train(:,1:end-1);       
train_label=train(:,end);
test_img=test(:,1:end-1);
test_label=test(:,end);

%把数据重置成四维矩阵并保存为mat形式，然后读入python进行处理
train_sz=size(train_img);
train_img_rz=reshape(train_img,[train_sz(1),64,64,3]);
save('train_img.mat','train_img_rz');


test_sz=size(test_img);
test_img_rz=reshape(test_img,[test_sz(1),64,64,3]);
save('test_img.mat','test_img_rz');

%把数字标签转换为码表形式的。
train_label_sz=size(train_label);          %大小
train_labels=zeros(train_label_sz(1),6);   %建立空label表
for i=1:train_label_sz(1)
    train_labels(i,train_label(i))=1;
end
save('train_label.mat','train_labels');

test_label_sz=size(test_label);          %大小
test_labels=zeros(test_label_sz(1),6);   %建立空label表
for i=1:test_label_sz(1)
    test_labels(i,test_label(i))=1;
end
save('test_label.mat','test_labels');
```
#3.2CNN框架及其训练。
经过上面的训练数据制作，我们一共得到了4个mat，分别是：
`train_img.mat, train_label.mat, test_img.mat,test_label.mat`
存储的分别是训练数据，训练标签，测试数据，测试标签。
![](https://upload-images.jianshu.io/upload_images/5252065-c7083c6323386ccb.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
我这里是两份，一份是乱序的，一份是没有乱序的。训练的时候直接使用的是乱序的，这个数据量大小一般，4个文件在400m左右。
CNN框架采用的比较简单，因为只是做一个6分类，框架如下：
![框架](https://upload-images.jianshu.io/upload_images/5252065-e3fb04ea013e1256.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
所有激活层的激活函数采用的都是relu()，权重初始化使用的是截断的高斯分布，偏置初始化卷积层为0，全连接层为0.1，训练是，全连接层采用0.5的dropout。
损失函数采用的是交叉熵损失：[交叉熵损失函数](https://blog.csdn.net/u012162613/article/details/44239919),主要是要避免采用平方损失带来的神经元（以sigmoid为例）参数更新过慢的问题。
优化器采用`tf.train.AdamOptimizer()`
训练时batch_size为100，只训练了一个epoch就达到了比较好的准确率（0.99-1），所以就没有训练过长时间。
值得注意的是，训练在加载训练数据之前，对数据进行归一化和中心化是必要的，我这里采用的比较简单：
```
train_img, train_label, test_img, test_label = get_date_and_label()

# 数据归一化，统一/255然后减掉0.5
train_img =np.float32(train_img) / 255 - 0.5
test_img = np.float32(test_img) / 255 - 0.5
```
训练过程就不说了，GTX1060 6G i57500训练是非常快的，总耗时20s左右，下午用笔记本试了一下则特别慢，每50个batch耗时1秒多，但是可能加载数据比较慢，实际耗时远大于这个。不过上个厕所也差不多训练好了。
训练好的模型保存起来待用就可以了，关于模型保存和使用这里有一篇博客我觉得讲的很清楚，贴在[这里](https://blog.csdn.net/marsjhao/article/details/72829635)。
#四.识别。
先定位，再识别，于获取人脸那里是差不多的，先定位人脸，然后取出来，resize成64_64的，作为输入输入placeholder，drop设置为1，predictions作为输出，对应可得到分类的信息，虽然图像是三维的，但是在输入的时候要把batch的这一维加上。然后根据分类信息画框图显示就可以了，进行了简单的测试，基本可以实现识别功能，偶有识别不出人脸(其实这个主要是因为harr人脸定位的原因)，还有一个就是偶尔会把没出现过的人脸误认为是已知的5类中的。主要原因还是在采集图像的过程中大家的表情还是都单一了一些。





