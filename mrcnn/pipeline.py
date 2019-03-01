from base_pipeline import BasePipeline
from mrcnn import model as modellib
from mrcnn.config import Config
from mrcnn.common_utils.lungmap_dataset import LungmapDataSet
from mrcnn.utils import get_file_from_remote
import os
import logging
import numpy as np
# weird import style to un-confuse PyCharm
try:
    from cv2 import cv2
except ImportError:
    import cv2

from mrcnn.common_utils.utils import display_instances

# FORMAT = '%(asctime)-15s %(message)s'
# logging.basicConfig(format=FORMAT, level=logging.DEBUG)


class LungMapTrainingConfig(Config):
    """
    Configuration for training
    """
    NAME = 'lungmap'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Number of classification classes (including background)
    # TODO: update this to be automatic
    NUM_CLASSES = 3  # Override in sub-classes
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)  # anchor side in pixels
    STEPS_PER_EPOCH = 200
    TRAIN_ROIS_PER_IMAGE = 300
    VALIDATION_STEPS = 5
    IMAGE_MAX_DIM = 256
    IMAGE_MIN_DIM = 256
    USE_MINI_MASK=False


config = LungMapTrainingConfig()


class MrCNNPipeline(BasePipeline):
    def __init__(self, image_set_dir, model_dir='mrcnn', test_img_index=0):
        super(MrCNNPipeline, self).__init__(image_set_dir, test_img_index)

        # TODO: make sure there are enough images in the image set, as this pipeline
        # needs to separate an image for validation, in addition to the reserved test image.
        self.model_dir = model_dir
        training_data = self.training_data.copy()
        test_data = {}
        validation_data = {}
        test_data[self.test_img_name] = training_data.pop(self.test_img_name)
        val_img_name = sorted(training_data.keys())[0]
        validation_data[val_img_name] = training_data.pop(val_img_name)

        self.dataset_train = LungmapDataSet()
        self.dataset_train.load_data_set(training_data)
        self.dataset_train.prepare()

        self.dataset_validation = LungmapDataSet()
        self.dataset_validation.load_data_set(validation_data)
        self.dataset_validation.prepare()

        self.dataset_test = LungmapDataSet()
        self.dataset_test.load_data_set(test_data)
        self.dataset_test.prepare()

        mask = self.dataset_validation.load_mask(0)
        self.model = modellib.MaskRCNN(
            mode="training",
            config=config,
            model_dir=self.model_dir
        )

    def train(self, coco_model_weights='mrcnn/mask_rcnn_coco.h5'):
        """
        Method to train new algorithm
        :param coco_model_weights: required pre-trained h5 file containing coco model weights
        :return:
        """
        if not os.path.exists(coco_model_weights):
            logging.info(
                'Could not find pre-trained weights maks_rcnn_coco_h5, caching now.'
            )
            get_file_from_remote('mrcnn', 'mask_rcnn_coco.h5')
            logging.info(
                'weights now cached: tmp/models/mrcnn/mask_rcnn_coco.h5'
            )

        self.model.load_weights(
            coco_model_weights,
            by_name=True,
            exclude=[
                "mrcnn_class_logits",
                "mrcnn_bbox_fc",
                "mrcnn_bbox",
                "mrcnn_mask"
            ]
        )
        self.model.train(
            self.dataset_train,
            self.dataset_validation,
            learning_rate=config.LEARNING_RATE,
            epochs=1,
            layers='heads'
        )
        self.model.train(
            self.dataset_train,
            self.dataset_validation,
            learning_rate=config.LEARNING_RATE / 10,
            epochs=2,
            layers="all"
        )

    def _convert_to_contours(self, maskrcnn_dict):
        """
        An internal
        :param maskrcnn_dict:
        :return:
        """
        masks = maskrcnn_dict[0]['masks']
        probs = maskrcnn_dict[0]['scores']
        class_names = [self.dataset_train.class_names[x] for x in maskrcnn_dict[0]['class_ids']]
        final = []
        for i in range(masks.shape[2]):
            instance = {'prob': {class_names[i]: probs[i]}}
            image_8bit = np.uint8(masks[:, :, i].astype('int'))
            _, contours, hierarchy = cv2.findContours(
                image_8bit,
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_SIMPLE
            )
            if len(contours) > 1:
                print('Returning more than 1 contour for a mask, what to do?')
                continue
            else:
                instance['contour'] = contours[0]
            final.append(instance)
        return final

    def test(self, model_weights='mask_rcnn_lungmap_0002.h5'):
        if not os.path.join(self.model_dir, model_weights):
            logging.info(
                'Could not find pre-trained weights mask_rcnn_lungmap_0002.h5, caching now.'
            )
            get_file_from_remote('mrcnn', model_weights)
            logging.info(
                'weights now cached: tmp/models/mrcnn/mask_rcnn_lungmap_0002.h5'
            )
        img = self.dataset_test.image_info[0]['img']
        model = modellib.MaskRCNN(
            mode="inference",
            config=config,
            model_dir=self.model_dir
        )
        model.load_weights(
            os.path.join(model_weights),
            by_name=True
        )
        a = model.detect([img], verbose=1)
        display_instances(
            image=img,
            boxes=a[0]['rois'],
            masks=a[0]['masks'],
            class_ids=a[0]['class_ids'],
            class_names=self.dataset_validation.class_names,
            scores=a[0]['scores']
        )

        truth = self.dataset_test.load_mask(0, with_bb=True)

        display_instances(
            image=img,
            boxes=truth[2],
            masks=truth[0],
            class_ids=truth[1],
            class_names=self.dataset_validation.class_names,
            scores=np.ones(truth[1].shape[0])
        )

        final = self._convert_to_contours(model.detect([img], verbose=1))
        self.test_results = final
