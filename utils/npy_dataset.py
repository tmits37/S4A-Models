import os
import json
import numpy as np
from typing import Tuple

from torch.utils.data import Dataset
from pycocotools.coco import COCO

from .settings.config import RANDOM_SEED, BANDS, IMG_SIZE, REFERENCE_BAND, NORMALIZATION_DIV, LINEAR_ENCODER


# IMG_SIZE = 366
# REFERENCE_BAND = 'B2'
# NORMALIZATION_DIV = 10000

# # --- For the selected classes, uncomment this section ---
# SELECTED_CLASSES = [
#     110,   # 'Wheat'
#     120,   # 'Maize'
#     140,   # 'Sorghum'
#     150,   # 'Barley'
#     160,   # 'Rye'
#     170,   # 'Oats'
#     330,   # 'Grapes'
#     435,   # 'Rapeseed'
#     438,   # 'Sunflower'
#     510,   # 'Potatoes'
#     770,   # 'Peas'
# ]

# LINEAR_ENCODER = {val: i + 1 for i, val in enumerate(sorted(SELECTED_CLASSES))}
# LINEAR_ENCODER[0] = 0
def min_max_normalize(image, percentile=2):
    image = image.astype('float32')

    percent_min = np.percentile(image, percentile, axis=(0,1))
    percent_max = np.percentile(image, 100-percentile, axis=(0,1))

    mask = np.mean(image, axis=2) != 0
    if image.shape[1] * image.shape[0] - np.sum(mask) > 0:
        mdata = np.ma.masked_equal(image, 0, copy=False)
        mdata = np.ma.filled(mdata, np.nan)
        percent_min = np.nanpercentile(mdata, percentile, axis=(0, 1))

    norm = (image-percent_min) / (percent_max - percent_min)
    norm[norm<0] = 0
    norm[norm>1] = 1
    norm = norm * mask[:,:,np.newaxis]
    # norm = (norm * 255).astype('uint8') * mask[:,:,np.newaxis]

    return norm

class SelectRandomStep(object):
    def __init__(self, start_month, end_month, delta_month=1, ratio=0.5):
        self.start_month = start_month
        self.end_month = end_month
        self.delta_month = delta_month
        self.ratio = ratio

    def __call__(self, x):
        # x.shape T, C, H, W
        idx = np.random.random()

        if idx > self.ratio:
            x = x[self.start_month:self.end_month]

        else:
            idx2 = np.random.random()
            if idx2 > 0.5:
                x = x[self.start_month-self.delta_month:self.end_month-self.delta_month]
            else:
                x = x[self.start_month+self.delta_month:self.end_month+self.delta_month]
        return x

class RandomCrop(object):
    """샘플데이터를 무작위로 자릅니다.

    Args:
        output_size (tuple or int): 줄이고자 하는 크기입니다.
                        int라면, 정사각형으로 나올 것 입니다.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, img, ann):
        image, landmarks = img, ann

        _, _, h, w = image.shape
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[:, :, top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks[top: top + new_h, left: left + new_w]
        img = image
        ann = landmarks

        return img, ann


class NpyPADDataset(Dataset):
    '''
        ├── dataset
        │   ├── my_dataset
        │   │   ├── scenario1_filename.json (Catalonia(2019, 2020), France(2019) -> Catalonia(2019, 2020), France(2019))
        │   │   ├── scenario2_filename.json (Catalonia(2019, 2020) -> France(2019)) 
        │   │   ├── scenario3_filename.json (Catalonia(2019), France(2019) -> Catalonia(2020))
        │   │   ├── ...
        │   │   ├── nrgb (B02 B03 B04 B08)
        │   │   │   ├── xxx{img_suffix}
        │   │   │   ├── yyy{img_suffix}
        │   │   │   ├── zzz{img_suffix}
        │   │   ├── rdeg (B05, B06, B07, B8A, B11, B12)
        │   │   │   ├── xxx{img_suffix}
        │   │   │   ├── yyy{img_suffix}
        │   │   │   ├── zzz{img_suffix}
        │   │   ├── label
        │   │   │   ├── xxx{seg_map_suffix}
        │   │   │   ├── yyy{seg_map_suffix}
        │   │   │   ├── zzz{seg_map_suffix}
    '''

    def __init__(
            self,
            root_dir: str = None,
            img_dir: str = None,
            ann_dir: str = None,
            band_mode: str ='nrgb',
            bands: list = None,
            linear_encoder: dict = None,
            start_month: int = 4,
            end_month: int = 10,
            output_size: tuple = (64,64),
            binary_labels: bool = False,
            return_parcels: bool = False,
            mode: str = 'test',
            scenario: int = 1,
            min_max_normalize: bool = True,
    ) -> None:
        '''
        Args:
            bands: list of str, default None
                A list of the bands to use. If None, then all available bands are
                taken into consideration. Note that the bands are given in a two-digit
                format, e.g. '01', '02', '8A', etc.
            linear_encoder: dict, default None
                Maps arbitrary crop_ids to range 0-len(unique(crop_id)).
            output_size: tuple of int, default None
                If a tuple (H, W) is given, then the output images will be divided
                into non-overlapping subpatches of size (H, W). Otherwise, the images
                will retain their original size.
            binary_labels: bool, default False
                Map categories to 0 background, 1 parcel.
            mode: str, ['train', 'val', 'test']
                The running mode. Used to determine the correct path for the median files.
            return_parcels: boolean, default False
                If True, then a boolean mask for the parcels is also returned.
        '''

        self.band_mode = band_mode
        if band_mode == 'nrgb':
            self.bands = ['B02', 'B03', 'B04', 'B08']

        elif band_mode == 'rdeg':
            self.bands = ['B02', 'B03', 'B04', 'B08', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12']
        
        else:
            raise RuntimeError

        self.return_parcels = return_parcels
        self.binary_labels = binary_labels

        if output_size is None:
            output_size = [IMG_SIZE, IMG_SIZE]
        assert isinstance(output_size[0], int) and isinstance(output_size[1], int),\
            'sub-patches dims must be integers!'
        assert output_size[0] == output_size[1], \
            f'Only square sub-patch size is supported. Mismatch: {output_size[0]} != {output_size[1]}.'
        self.output_size = [int(dim) for dim in output_size]

        # We index based on year, number of bins should be the same for every year
        # therefore, calculate them using a random year

        self.img_dir = img_dir if img_dir else os.path.join(root_dir, 'nrgb')
        self.ann_dir = ann_dir if ann_dir else os.path.join(root_dir, 'label')
        self.root_dir = root_dir if root_dir else os.path.dirname(self.img_dir)

        self.start_month = start_month - 1
        self.end_month = end_month - 1
        self.linear_encoder = LINEAR_ENCODER
        self.min_max_normalize = min_max_normalize

        self.mode = mode
        self.scenario = scenario
        assert self.mode in ['train', 'val', 'test'], \
            "variable mode should be 'train' or 'val' or 'test'"
        assert self.scenario == 1 or self.scenario == 2 or self.scenario == 3, \
            'variable scenario should be 1 or 2 or 3.'

        with open(os.path.join(self.root_dir, f"scenario{self.scenario}_filename.json"), "r") as st_json:
            self.img_infos = json.load(st_json)[mode]
            self.img_infos = [f + '.npy' for f in self.img_infos]

        if output_size[0] != IMG_SIZE:
            self.transforms = RandomCrop(output_size[0])
        else:
            self.transforms = None

        # self.select_randomstep = SelectRandomStep(self.start_month, self.end_month, delta_month=1)

        print('Rootdir: {}'.format(self.root_dir))
        print('Scenario: {}, MODE: {}, length of datasets: {}'.format(self.scenario, self.mode, len(self.img_infos)))
        print('Acquired Data Month: From {} to {}'.format(self.start_month + 1, self.end_month + 1))
        print(f'Data shape: [T, C, H, W] ({self.end_month - self.start_month}, {len(self.bands)}, {output_size[0]}, {output_size[0]})')


    def prepare_train_img(self, idx: int) -> dict:
        if self.band_mode == 'nrgb':
            readpath = os.path.join(self.img_dir, self.img_infos[idx])
            img = np.load(readpath)
        elif self.band_mode == 'rdeg':
            readpath = os.path.join(self.img_dir, self.img_infos[idx])
            img = np.load(readpath)

            rdegpath = os.path.join(os.path.dirname(self.img_dir), 'rdeg', self.img_infos[idx])
            rdeg = np.load(rdegpath)
            img = np.stack([img, rdeg], axis=1)
        else:
            raise RuntimeError

        annpath = os.path.join(self.ann_dir, self.img_infos[idx])
        ann = np.load(annpath)

        # if self.mode == 'train':
        #     img = self.select_randomstep(img)
        # else:
        #     img = img[self.start_month:self.end_month]
        img = img[self.start_month:self.end_month]

        if self.transforms:
            img, ann = self.transforms(img, ann)

        return img, ann

    def _normalize(self, img):
        if self.min_max_normalize:
            T, C, H, W = img.shape
            img = img.reshape(T*C, H, W)
            img = min_max_normalize(img.transpose(1,2,0), percentile=0.5)
            img = img.transpose(2,0,1)
            img = img.reshape(T, C, H, W)
        else:
            img = np.divide(img, NORMALIZATION_DIV) #  / 10000
        return img
    
    def __getitem__(self, idx: int) -> dict:
        img, ann = self.prepare_train_img(idx)

        # Normalize data to range [0-1]
        img = self._normalize(img)

        out = {}
        if self.return_parcels:
            parcels = ann != 0
            out['parcels'] = parcels

        if self.binary_labels:
            # Map 0: background class, 1: parcel
            ann[ann != 0] = 1
        else:
            # Map labels to 0-len(unique(crop_id)) see config
            # labels = np.vectorize(self.linear_encoder.get)(labels)
            _ = np.zeros_like(ann)
            for crop_id, linear_id in self.linear_encoder.items():
                _[ann == crop_id] = linear_id
            ann = _

        # # Map all classes NOT in linear encoder's values to 0
        ann[~np.isin(ann, list(self.linear_encoder.values()))] = 0

        out['medians'] = img.astype(np.float32)
        out['labels'] = ann.astype(np.int64)

        return out


    def __len__(self):
        return len(self.img_infos)


if __name__ == '__main__':
    rootdir = '/nas/k8s/dev/research/doyoungi/croptype_cls/S4A-Models/dataset/newdata/'
    dataset =  NpyPADDataset(root_dir=rootdir, band_mode='nrgb', start_month=4)
    print(dataset[0])
