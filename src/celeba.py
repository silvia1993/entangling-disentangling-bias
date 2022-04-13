import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image

class CelebaDataset(Dataset):
    def __init__(self, list_IDs, labels, transform=T.ToTensor()):
        self.labels = labels
        self.list_IDs = list_IDs
        self.transform = transform

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        img = Image.open(ID)
        X = self.transform(img)
        y = self.labels[ID]

        return X,y[0],y[1],y


def create_dataset(batch_size,path, attribute, protected_attribute, augment, number=0, split='train'):

    img_path = path+'/img_align_celeba/'
    attr_path = path+'/list_attr_celeba.txt'
    list_ids = []
    label = open(attr_path, 'r')
    label = label.readlines()
    train_beg = 0
    valid_beg = 162770
    test_beg = 182637

    if split=='train':
        if number==0:
            number = valid_beg - train_beg
        beg = train_beg
    elif split=='valid':
        if number==0:
            number = test_beg - valid_beg
        beg = valid_beg
    elif split=='test':
        if number==0:
            number = 202599 - test_beg
        beg = test_beg
    else:
        print('Error')
        return
    attr = {}
    for i in range(beg+2, beg+ number+2):
        temp = label[i].strip().split()
        list_ids.append(img_path+temp[0])
        attr[img_path+temp[0]]=torch.Tensor([int((int(temp[attribute+1])+1)/2),  int((int(temp[protected_attribute+1])+1)/2)])

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

    if augment:
        transform = T.Compose([
            T.Resize(64),
            T.Resize(256),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize
        ])
    else:
        transform = T.Compose([
            T.Resize(64),
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            normalize
        ])

    #print(transform)
    dset = CelebaDataset(list_ids, attr, transform)

    loader = DataLoader(dataset=dset,
                                 batch_size=batch_size,
                                 shuffle=augment,
                                 num_workers=4,
                                 pin_memory=True)

    return loader

