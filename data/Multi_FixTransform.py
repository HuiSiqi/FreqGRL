#modified from https://github.com/facebookresearch/swav/blob/master/src/multicropdataset.py
import torch
from PIL import Image
from torchvision import transforms
from data.Image_ops import GaussianBlur
from data.RandAugment import RandAugment,RandAugmentPikey
import data.additional_transforms as add_transforms

class Multi_Fixtransform(object):
    def __init__(self,
            size_crops,
            nmb_crops,
            min_scale_crops,
            max_scale_crops,
            aug_times,init_size=224):
        """
        :param size_crops: list of crops with crop output img size
        :param nmb_crops: number of output cropped image
        :param min_scale_crops: minimum scale for corresponding crop
        :param max_scale_crops: maximum scale for corresponding crop
        :param normalize: normalize operation
        :param aug_times: strong augmentation times
        :param init_size: key image size
        """
        normalize_param = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        jitter_param = dict(Brightness=0.4, Contrast=0.4, Color=0.4)
        self.image_size = init_size
        self.normalize_param = normalize_param
        self.normzalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        self.jitter_param = jitter_param

        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)

        trans=[]
        #key image transform
        # self.weak = transforms.Compose([
        #     transforms.RandomResizedCrop(init_size, scale=(0.2, 1.)),
        #     transforms.RandomApply([
        #         transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        #     ], p=0.8),
        #     transforms.RandomGrayscale(p=0.2),
        #     transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     self.normalize
        # ])

        transform_list = ['RandomResizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        transform_funcs = [self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        trans.append(transform)
        self.aug_times=aug_times
        # trans_weak=[]
        trans_strong=[]
        for i in range(len(size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )

            strong = transforms.Compose([
            randomresizedcrop,
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            RandAugmentPikey(n=self.aug_times, m=10),
            transforms.ToTensor(),
            self.normzalize
            ])
            # weak=transforms.Compose([
            # randomresizedcrop,
            # transforms.RandomApply([
            #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            # ], p=0.8),
            # transforms.RandomGrayscale(p=0.2),
            # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            # transforms.RandomHorizontalFlip(),
            # transforms.ToTensor(),
            # normalize
            # ])
            # trans_weak.extend([weak]*nmb_crops[i])
            trans_strong.extend([strong]*nmb_crops[i])
        # trans.extend(trans_weak)
        trans.extend(trans_strong)
        self.trans=trans

    def __call__(self, x):
        multi_crops = list(map(lambda trans: trans(x), self.trans))
        return multi_crops

    def parse_transform(self, transform_type):
        if transform_type == 'ImageJitter':
            method = add_transforms.ImageJitter(self.jitter_param)
            return method
        method = getattr(transforms, transform_type)

        if transform_type == 'RandomResizedCrop':
            return method(self.image_size)
        elif transform_type == 'CenterCrop':
            return method(self.image_size)
        elif transform_type == 'Resize':
            return method([int(self.image_size * 1.15), int(self.image_size * 1.15)])
        elif transform_type == 'Normalize':
            return method(**self.normalize_param)
        else:
            return method()

    def get_composed_transform(self, aug=False):
        if aug:
            return self
        else:
            transform_list = ['Resize', 'CenterCrop', 'ToTensor', 'Normalize']
        transform_funcs = [self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform
    
if __name__ == '__main__':
    size_crops = [224]
    nmb_crops = [1]
    min_scale_crops=[0.2]
    max_scale_crops=[1.0]
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    aug_times=5
    fix_transform = Multi_Fixtransform(size_crops,
                                       nmb_crops,
                                       min_scale_crops,
                                       max_scale_crops, aug_times)
    # fix_transform = Multi_Fixtransform
    image = Image.open('/data/pikey/code/FSL/CDFSL/Adversarial/analyse/DemoTask/query1/Geococcyx_0040_104507.jpg')
    for i in range(10):
        o = fix_transform(image)
        import torchvision
        torchvision.utils.save_image(o[0],f'/data/pikey/code/FSL/CDFSL/Adversarial/analyse/DemoTask/query1/weak_augment{i}.png',nrow=1,normalize=True,padding=0)
        torchvision.utils.save_image(o[1],f'/data/pikey/code/FSL/CDFSL/Adversarial/analyse/DemoTask/query1/strong_augment{i}.png',nrow=1,normalize=True,padding=0)
    # torchvision.utils.save_image(o[1],'strong.png',nrow=1,normalize=True,padding=0)