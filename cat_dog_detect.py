import os
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

dataset_dir = '/openbayes/input/input0/data/data'
# 设置测试集和训练集存放的目录
train_dir = '/openbayes/input/input0/data/train_data'
test_dir = '/openbayes/input/input0/data/test_data'

# 创建训练集和测试集的目录
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(os.path.join(train_dir, 'cats'), exist_ok=True)
os.makedirs(os.path.join(train_dir, 'dogs'), exist_ok=True)
os.makedirs(os.path.join(test_dir, 'cats'), exist_ok=True)
os.makedirs(os.path.join(test_dir, 'dogs'), exist_ok=True)

# 获取数据集中所有文件的文件名
all_files = os.listdir(dataset_dir)

# 根据文件名将文件分为猫和狗的列表
cats = [f for f in all_files if 'cat' in f.lower()]
dogs = [f for f in all_files if 'dog' in f.lower()]

# 以8:2的比例划分测试集和训练集
train_cats, test_cats = train_test_split(cats, test_size=0.2, random_state = 42)
train_dogs, test_dogs = train_test_split(dogs, test_size=0.2, random_state = 42)

def copy_files(file_list, source_dir, target_dir):
    for file_name in file_list:
        shutil.copy(os.path.join(source_dir, file_name), os.path.join(target_dir, file_name))

# 复制猫的图片
copy_files(train_cats, dataset_dir, os.path.join(train_dir, 'cats'))
copy_files(test_cats, dataset_dir, os.path.join(test_dir, 'cats'))

# 复制狗的图片
copy_files(train_dogs, dataset_dir, os.path.join(train_dir, 'dogs'))
copy_files(test_dogs, dataset_dir, os.path.join(test_dir, 'dogs'))

# 查看训练集和测试集中猫狗划分的数量
print(f'训练集中猫的图片数量: {len(os.listdir(os.path.join(train_dir, "cats")))}')
print(f'训练集中狗的图片数量: {len(os.listdir(os.path.join(train_dir, "dogs")))}')
print(f'测试集中猫的图片数量: {len(os.listdir(os.path.join(test_dir, "cats")))}')
print(f'测试集中狗的图片数量: {len(os.listdir(os.path.join(test_dir, "dogs")))}')

def define_transfer_model():
    # 构建不带分类器的预训练模型
    base_model = InceptionV3(weights='imagenet', include_top=False)
    
    # 添加全局平均池化层
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # 添加一个全连接层
    x = Dense(1024, activation='relu')(x)
    
    # 添加一个分类器
    prediction = Dense(1, activation='sigmoid')(x)
    
    # 锁住卷积层
    for layer in base_model.layers:
        layer.trainable = False
    model = Model(inputs=base_model.input, outputs=prediction)
    
    #编译模型
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model
    
def train_transfer_model():
    # 实例化模型
    model = define_transfer_model()
    # 创建图片生成器
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    train_it = datagen.flow_from_directory(
        train_dir,
        class_mode='binary',
        batch_size=64,
        target_size=(224,224)
    )
    # 训练模型
    model.fit_generator(train_it,
                        steps_per_epoch = train_it.samples // 64,
                        epochs=10,
                        verbose=1)
    return model

trained_model = train_transfer_model()

def evaluate_model(model):
    # 创建测试数据生成器
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    test_it = test_datagen.flow_from_directory(
        test_dir,
        class_mode='binary',
        batch_size=64,
        target_size=(224, 224),
        shuffle=False  # 关闭混洗以确保顺序一致
    )
    # 评估模型
    loss, accuracy = model.evaluate(test_it, verbose=1)
    print('Test accuracy:', accuracy)

# 评估模型
evaluate_model(trained_model)