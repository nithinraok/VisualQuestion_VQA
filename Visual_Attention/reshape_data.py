import h5py
file="/data/digbose92/VQA/COCO/train_hdf5_COCO/train_feats_resnet152_pool.hdf5"
data=h5py.File(file)
feats=data['feats']
print('Loading numpy weights')
feats_numpy=feats.value
data.close()
print('weights loaded')
feats_numpy=feats_numpy.reshape(feats_numpy.shape[0],2048,49)
feats_numpy=np.transpose(feats_numpy,(0,2,1))
print('transpose done')
write_hdf5=h5py.File(file,'w')
print('dataset writing done')
write_hdf5.create_dataset('feats',data=feats_numpy)
write_hdf5.close()

