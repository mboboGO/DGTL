import os
import scipy.io as io
import numpy as np
import h5py

def read_list(data_list):
    images = []
    fid = open(data_list,'r')
    for line in fid.readlines():
        data = line.strip('\n').split(' ')
        item=(data[0],np.int64(data[1]))
        images.append(item)
    fid.close()
    return images

def checkdir(datapath):
    if not os.path.exists(datapath):
        os.makedirs(datapath)
    return datapath
def checkfile(datapath):
    assert os.path.exists(datapath), 'This is no file %s'%(datapath)
    return datapath

def att_make(data_list):
    fine2coarse = dict()
    fine_labels = []
    coarse_labels = []
    n = 0
    for item in data_list:
        path = item[0]
        fine_name = path.split('/')[0].split('.')[-1]
        coarse_name = path.split('/')[0].split('_')[-1]
        if coarse_name not in coarse_labels:
            coarse_labels.append(coarse_name)
            n += 1
        if fine_name not in fine_labels:
            fine_labels.append(fine_name)
            fine2coarse[fine_name] = n-1
    return coarse_labels,fine_labels,fine2coarse

def save_cub_data():
    print('### Load CUB data')
    print('current path:',os.getcwd())
    ''' path setting '''
    split_path = '/userhome/raw_data/zsl_data/CUB'
    image_path = '/userhome/raw_data/CUB_200_2011/CUB_200_2011/images/'
    traindir = os.path.join(image_path,'../train.list')
    valdir = os.path.join(image_path,'../test.list')

    ''' make c and f data list '''
    att_data = io.loadmat(checkfile(os.path.join(split_path, 'att_splits.mat')))
    train_list_ = read_list(traindir)
    val_list_ = read_list(valdir)
    coarse_labels,fine_labels,fine2coarse = att_make(train_list_)
    nc = len(coarse_labels)
    nf = len(fine_labels)
    
    train_list = []
    for item_ in train_list_:
        path = item_[0]
        name = path.split('/')[0].split('.')[-1]
        item=(path,np.int64(fine2coarse[name]))
        train_list.append(item)
        
    val_list = []
    for item_ in val_list_:
        path = item_[0]
        name = path.split('/')[0].split('.')[-1]
        item=(path,np.int64(fine2coarse[name]),item_[1])
        val_list.append(item)
        
    ''' att '''
    fine_att = att_data['att'].transpose()
    coarse_att = np.zeros([nc,fine_att.shape[1]])
    num = np.zeros([nc,1])
    f2c = np.zeros([nf,1])
    for f_i,label in enumerate(fine_labels):
        c_i = fine2coarse[label]
        coarse_att[c_i] += fine_att[f_i,:]
        f2c[f_i,0]=c_i
        num[c_i,0]=+1
    coarse_att /= num
    print('coarse_att:',coarse_att.shape)
    print('fine_att:',fine_att.shape)
        
    ''' save '''
    save_path = checkdir(os.path.join('./cub'))
    h5_path = os.path.join(save_path, 'data_info.h5')

    if os.path.exists(h5_path):
        print("Skip store semantic features.")
    else:
        h5_semantic_file = h5py.File(h5_path, 'w')
        # save att
        h5_semantic_file.create_dataset('fine_att', fine_att.shape, dtype=np.float32)
        h5_semantic_file.create_dataset('coarse_att', coarse_att.shape, dtype=np.float32)
        h5_semantic_file.create_dataset('f2c', f2c.shape, dtype=np.int16)
        # image path

        h5_semantic_file['fine_att'][...] = fine_att
        h5_semantic_file['coarse_att'][...] = coarse_att
        h5_semantic_file['f2c'][...] = f2c
        h5_semantic_file['img_path'] = image_path

        h5_semantic_file.close()

    ''' write visual feats '''
    train_fid = open(save_path+'/train.list','w') 
    test_fid  = open(save_path+'/test.list','w')
    coarse_fid  = open(save_path+'/coarse_name.list','w')
    
    for item in train_list:
        train_fid.write('{} {} -1\n'.format(item[0],item[1]))
    train_fid.close()
    
    for item in val_list:
        test_fid.write('{} {} {}\n'.format(item[0],item[1],item[2]))
    test_fid.close()
    
    #for item in coarse_name:
    #    coarse_fid.write('{}\n'.format(coarse_name))
    #coarse_fid.close()
    

if __name__ == '__main__':
    save_cub_data()
