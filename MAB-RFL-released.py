import keras.datasets.cifar10
import tensorflow as tf
import numpy as np
import os
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,Input,Dropout
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.disable_eager_execution()
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.05
session = tf.compat.v1.Session(config=config)
print(tf.test.is_gpu_available())

import random
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import norm
from sklearn.decomposition import PCA
import networkx as nx

local_epoch = 3 #本地迭代次数
client_num = 50 #用户总数
global_epoch = 100  #全局迭代次数
f = 15   #攻击者数目
miu = 0.1
c_max = 0.3
c_min = 0.1
alpha = 0

(train_images, train_labels) ,(test_images, test_labels) = keras.datasets.cifar10.load_data()
train_images= train_images/255.0
test_images= test_images/255.0

def getById(id, train_images, train_labels):
    resImg = []
    resLab = []
    for i in range(len(train_labels)):
        if train_labels[i] == id:
            resImg.append(train_images[i])
            resLab.append(id)
    return resImg,resLab

def ExcludeNum(n):  #用于算非dominated标签集合
    res = list(range(0,10))
    res.remove(n)
    return res
domi_label_conut = []
dataset_size_list = []
def NIID(degree, clientNum):    #Non-IID数据划分
    imagesGroup = []
    labelsGroup = []
    resImg = []
    resLab = []
    for i in range(10):
        tempimg, templab = getById(i, train_images, train_labels)
        imagesGroup.append(tempimg)
        labelsGroup.append(templab)
    Index = [0]*10
    for i in range(clientNum):
        tempimg = []
        templab = []
        dominated_label = random.randint(0,9)
        domi_label_conut.append(dominated_label)
        non_dominated = ExcludeNum(dominated_label)
        dataset_size = random.randint(1200,2000)
        dataset_size_list.append(dataset_size)
        for j in range(dataset_size): #每个用户的训练集大小在1200到2000之间随机生成
            randnum = random.random()
            selected_label = -1
            if(randnum<degree):
                selected_label = dominated_label
            else:
                selected_label = random.choice(non_dominated)
            rand = np.random.randint(0,len(imagesGroup[selected_label])-1)
            tempimg.append(imagesGroup[selected_label][rand])
            templab.append(labelsGroup[selected_label][rand])
            Index[selected_label]+=1
        index_ = [k for k in range(len(tempimg))]
        np.random.shuffle(index_)
        tempimg = np.array(tempimg)[index_]
        templab = np.array(templab)[index_]
        resImg.append(tempimg)
        resLab.append(templab)
    print([domi_label_conut.count(i) for i in range(0,10)])
    print(domi_label_conut)
    print(dataset_size_list)
    return resImg, resLab

image,label = NIID(0.5,client_num)
print(label)


def getSTD(modelweights):
    std = []
    for i in range(len(InitialModel_weight)):
        std.append(np.std([modelweights[j][i] for j in range(len(modelweights))],axis=0))
    return np.array(std)
def VGG():
    x = Input(shape=(32, 32, 3))
    y = x
    y = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu',
                      kernel_initializer='he_normal')(y)
    y = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu',
                      kernel_initializer='he_normal')(y)
    y = MaxPooling2D(pool_size=2, strides=2, padding='same')(y)

    y = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu',
                      kernel_initializer='he_normal')(y)
    y = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu',
                      kernel_initializer='he_normal')(y)
    y = MaxPooling2D(pool_size=2, strides=2, padding='same')(y)

    y = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu',
                      kernel_initializer='he_normal')(y)
    y = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu',
                      kernel_initializer='he_normal')(y)
    y = MaxPooling2D(pool_size=2, strides=2, padding='same')(y)

    y = Flatten()(y)

    y = Dropout(0.5)(y)

    y = Dense(units=10, activation='softmax', kernel_initializer='he_normal')(y)
    model = Model(inputs=x, outputs=y, name='model1')
    return model
class Client:   #用户
    def __init__(self,trainImages,trainLabels,id):
        self.trainImages = trainImages
        self.trainLabels = trainLabels
        self.id = id
        self.trainSetNum = len(trainImages)
        self.model = VGG()
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.byzantine=False    #是否是拜占庭用户
        self.succ = 1
        self.fail = 1
        self.momentum = np.array(self.model.get_weights())*0
        self.update = np.array(self.model.get_weights())*0
        self.smtcos = 0
        self.seleted_epoch = 0
    def broadcast(self,globalModelWeight):  #中心服务器将全局模型广播给用户
        self.model.set_weights(globalModelWeight)
    def train(self):    #本地训练
        self.model.fit(self.trainImages, self.trainLabels, batch_size=64, epochs=local_epoch)
        self.update = np.array(self.model.get_weights())-np.array(globalModel_weight)
    def setUpdate(self,new_update):
        self.update = new_update

def flat(v):    #把模型拉升为1维向量
    t=[]
    for i in range(len(v)):
        temp = np.array(v[i]).reshape(-1)
        t.extend(temp)
    return t

def EuclideanDistance(modelweight1,modelweight2):
    list1=np.array(modelweight1)
    list2=np.array(modelweight2)
    list3=np.square(list1-list2)
    sum=[]
    for l in list3:
        for c in l:
            sum.append(np.sum(np.sum(c)))
    return np.sqrt(np.sum(sum))

def Cos_Similarity(modelweight1, modelweight2,unit):
    multi = np.array(modelweight1)*np.array(modelweight2)
    sum = []
    for l in multi:
        for c in l:
            sum.append(np.sum(c))
    return min(np.sum(sum),1) if unit else min(np.sum(sum)/(getWeightNorm(modelweight1)*getWeightNorm(modelweight2)),1)

def getWeightNorm(modelweight):
    list = np.square(modelweight)
    sum = []
    for l in list:
        for c in l:
            sum.append(np.sum(np.sum(c)))
    return np.sqrt(np.sum(sum))

def LIE(clients,selectedId):
    benignId = []
    for i in selectedId:
        if clients[i].byzantine==False:
            benignId.append(i)
    s = int(client_num/2+1)-f
    z = norm.ppf((client_num-f-s)/(client_num-f))
    mean = np.mean([clients[i].update for i in benignId],axis=0)
    std = getSTD([clients[i].update for i in benignId])
    bad_update = mean-z*std
    for i in selectedId:
        if clients[i].byzantine==True:
            clients[i].update = bad_update
    return

def TS(clients):
    selectedId = []
    for i in range(client_num):
        p = np.random.beta(clients[i].succ,clients[i].fail) #每个用户被选中的概率
        if p>=0.9:
            selectedId.append(clients[i].id)
        if p>0.2 and p<=0.9 and np.random.random()<p :
            selectedId.append(clients[i].id)
    if len(selectedId) == 0:       #如果本轮采样结果为空，则所有用户均被选中
        selectedId = [i for i in range(client_num)]
    print("本轮被选中用户为：",selectedId)
    return selectedId
def MAB_FL(clients, selectedId, epoch):
    for i in selectedId:       #把所有被选中的用户的动量单位化
        clients[i].momentum = clients[i].update+miu**(epoch-clients[i].seleted_epoch)*clients[i].momentum
        clients[i].momentum = clients[i].momentum/getWeightNorm((np.array(clients[i].momentum)))
        clients[i].seleted_epoch = epoch
    # 剔除女巫
    G = nx.Graph()
    edges = []
    for i in range(len(selectedId)):
        for j in range(i+1,len(selectedId)):
            if Cos_Similarity(clients[selectedId[i]].momentum,clients[selectedId[j]].momentum,True)>max(c_max*np.exp(-epoch/20),c_min):
                edges.append((selectedId[i],selectedId[j]))
    G.add_nodes_from(selectedId)
    G.add_edges_from(edges)
    C = sorted(nx.connected_components(G), key=len, reverse=True)
    print(C)
    print(C[0]) #最大连通子图
    if (len(C[0])>1):
        for i in C[0]:
            clients[i].fail += 1
            selectedId.remove(i)
    print(selectedId)
    local_updates = [flat(clients[i].momentum)for i in selectedId]
    pca = PCA(n_components=0.95)
    X_reduced = pca.fit_transform(local_updates)
    print(X_reduced)
    estimator = AgglomerativeClustering(2)  # 构造聚类器
    estimator.fit(X_reduced)  # 聚类
    label_pred = estimator.labels_  # 获取聚类标签
    selectedId_c1 = []
    selectedId_c2 = []
    for i in range(len(selectedId)):
        if label_pred[i]==0:
            selectedId_c1.append(selectedId[i])
        else:
            selectedId_c2.append(selectedId[i])
    print(selectedId_c1)
    print(selectedId_c2)
    print(label_pred)
    m1 = np.mean([clients[i].momentum for i in selectedId_c1],axis=0)
    m2 = np.mean([clients[i].momentum for i in selectedId_c2],axis=0)
    cos_between_clusters = Cos_Similarity(m1,m2,False)
    print(cos_between_clusters)
    if cos_between_clusters<alpha:
        if len(selectedId_c1)>len(selectedId_c2):
            for i in selectedId_c2:
                clients[i].fail += 1
                selectedId.remove(i)
        else:
            for i in selectedId_c1:
                clients[i].fail += 1
                selectedId.remove(i)
    for i in selectedId:
        clients[i].succ+=1
    print("final aggregation:",selectedId)
    lr = np.median([getWeightNorm(clients[i].update) for i in selectedId])
    return lr,np.mean([clients[i].momentum for i in selectedId],axis=0)

InitialModel = VGG()
InitialModel.compile(optimizer='adadelta', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # 配置模型的一些参数
InitialModel_weight =  InitialModel.get_weights()

if __name__ == '__main__':
    model_accuracy_list = []
    model_loss_list = []
    globalModel = VGG()
    globalModel.compile(optimizer='adadelta', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # 配置模型的一些参数
    globalModel.summary()
    globalModel_weight = InitialModel_weight
    clients=[]
    for i in range(0, client_num):
        clients.append(Client(image[i], label[i],i))
        if i >= client_num - f:
            clients[i].byzantine = True
    for c in clients:
        c.broadcast(globalModel_weight)
    for epoch in range(global_epoch):
        if epoch<10: #前10轮选中所有用户
            selectedId = [i for i in range(client_num)]
        else:   #否则使用汤普森采样选择用户
            selectedId = TS(clients)
        for i in selectedId:
            if clients[i].byzantine == False:    #只有正常用户才训练
                print("training on client " + str(clients[i].id) + "(global epoch " + str(epoch + 1) + "/" + str(global_epoch) + ")")
                clients[i].train()
        LIE(clients, selectedId)  # 发动LIE攻击
        lr,gp = MAB_FL(clients,selectedId,epoch)
        global_update= gp
        print(lr)
        globalModel_weight = globalModel_weight+lr*global_update
        globalModel.set_weights(globalModel_weight)
        test_loss, test_acc = globalModel.evaluate(test_images, test_labels)
        print(test_acc)
        model_accuracy_list.append(float('%.4f' % test_acc))
        model_loss_list.append(float('%.4f' % test_loss))
        for c in clients:
            c.broadcast(globalModel_weight)
