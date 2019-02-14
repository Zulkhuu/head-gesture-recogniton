# IMPORTANT:
#
# This code is derivated from the MTCNN implementation of David Sandberg for Facenet
# (https://github.com/davidsandberg/facenet/)
# It has been rebuilt from scratch, taking the David Sandberg's implementation as a reference.
#

import cv2
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

#import pkg_resources
#from mtcnn.layer_factory import LayerFactory
#from mtcnn.network import Network
#from mtcnn.exceptions import InvalidImage

class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, c, h, w].
        Returns:
            a float tensor with shape [batch_size, c*h*w].
        """

        # without this pretrained model isn't working
        x = x.transpose(3, 2).contiguous()

        return x.view(x.size(0), -1)

class PNet(nn.Module):
    """
    Network to propose areas with faces.
    """
    def __init__(self):
        super(PNet, self).__init__()

        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 10, 3, 1)),
            ('prelu1', nn.PReLU(10)),
            ('pool1', nn.MaxPool2d(2, 2, ceil_mode=True)),

            ('conv2', nn.Conv2d(10, 16, 3, 1)),
            ('prelu2', nn.PReLU(16)),

            ('conv3', nn.Conv2d(16, 32, 3, 1)),
            ('prelu3', nn.PReLU(32))
        ]))

        self.conv4_1 = nn.Conv2d(32, 2, 1, 1)
        self.conv4_2 = nn.Conv2d(32, 4, 1, 1)

    def set_weights(self, weights_file):
        weights = np.load(weights_file)[()]
        for n, p in self.named_parameters():
            p.data = torch.FloatTensor(weights[n])

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, 3, h, w].
        Returns:
            b: a float tensor with shape [batch_size, 4, h', w'].
            a: a float tensor with shape [batch_size, 2, h', w'].
        """
        x = self.features(x)
        a = self.conv4_1(x)
        b = self.conv4_2(x)
        a = F.softmax(a)
        return b, a

class RNet(nn.Module):
    """
    Network to refine the areas proposed by PNet
    """
    def __init__(self):
        super(RNet, self).__init__()

        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 28, 3, 1)),
            ('prelu1', nn.PReLU(28)),
            ('pool1', nn.MaxPool2d(3, 2, ceil_mode=True)),

            ('conv2', nn.Conv2d(28, 48, 3, 1)),
            ('prelu2', nn.PReLU(48)),
            ('pool2', nn.MaxPool2d(3, 2, ceil_mode=True)),

            ('conv3', nn.Conv2d(48, 64, 2, 1)),
            ('prelu3', nn.PReLU(64)),

            ('flatten', Flatten()),
            ('conv4', nn.Linear(576, 128)),
            ('prelu4', nn.PReLU(128))
        ]))

        self.conv5_1 = nn.Linear(128, 2)
        self.conv5_2 = nn.Linear(128, 4)

    def set_weights(self, weights_file):
        weights = np.load(weights_file)[()]
        for n, p in self.named_parameters():
            p.data = torch.FloatTensor(weights[n])

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, 3, h, w].
        Returns:
            b: a float tensor with shape [batch_size, 4].
            a: a float tensor with shape [batch_size, 2].
        """
        x = self.features(x)
        a = self.conv5_1(x)
        b = self.conv5_2(x)
        a = F.softmax(a)
        return b, a

class ONet(nn.Module):
    """
    Network to retrieve the keypoints
    """
    def __init__(self):
        super(ONet, self).__init__()

        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 32, 3, 1)),
            ('prelu1', nn.PReLU(32)),
            ('pool1', nn.MaxPool2d(3, 2, ceil_mode=True)),

            ('conv2', nn.Conv2d(32, 64, 3, 1)),
            ('prelu2', nn.PReLU(64)),
            ('pool2', nn.MaxPool2d(3, 2, ceil_mode=True)),

            ('conv3', nn.Conv2d(64, 64, 3, 1)),
            ('prelu3', nn.PReLU(64)),
            ('pool3', nn.MaxPool2d(2, 2, ceil_mode=True)),

            ('conv4', nn.Conv2d(64, 128, 2, 1)),
            ('prelu4', nn.PReLU(128)),

            ('flatten', Flatten()),
            ('conv5', nn.Linear(1152, 256)),
            ('drop5', nn.Dropout(0.25)),
            ('prelu5', nn.PReLU(256)),
        ]))

        self.conv6_1 = nn.Linear(256, 2)
        self.conv6_2 = nn.Linear(256, 4)
        self.conv6_3 = nn.Linear(256, 10)

    def set_weights(self, weights_file):
        weights = np.load(weights_file)[()]
        for n, p in self.named_parameters():
            p.data = torch.FloatTensor(weights[n])

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [batch_size, 3, h, w].
        Returns:
            c: a float tensor with shape [batch_size, 10].
            b: a float tensor with shape [batch_size, 4].
            a: a float tensor with shape [batch_size, 2].
        """
        x = self.features(x)
        a = self.conv6_1(x)
        b = self.conv6_2(x)
        c = self.conv6_3(x)
        a = F.softmax(a)
        return c, b, a

class StageStatus(object):
    """
    Keeps status between MTCNN stages
    """
    def __init__(self, pad_result: tuple=None, width=0, height=0):
        self.width = width
        self.height = height
        self.dy = self.edy = self.dx = self.edx = self.y = self.ey = self.x = self.ex = self.tmpw = self.tmph = []

        if pad_result is not None:
            self.update(pad_result)

    def update(self, pad_result: tuple):
        s = self
        s.dy, s.edy, s.dx, s.edx, s.y, s.ey, s.x, s.ex, s.tmpw, s.tmph = pad_result

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MTCNN(object):
    """
    Allows to perform MTCNN Detection ->
        a) Detection of faces (with the confidence probability)
        b) Detection of keypoints (left eye, right eye, nose, mouth_left, mouth_right)
    """

    def __init__(self, min_face_size: int=20, steps_threshold: list=None,
                 scale_factor: float=0.709):
        """
        Initializes the MTCNN.
        :param weights_file: file uri with the weights of the P, R and O networks from MTCNN. By default it will load
        the ones bundled with the package.
        :param min_face_size: minimum size of the face to detect
        :param steps_threshold: step's thresholds values
        :param scale_factor: scale factor
        """
        if steps_threshold is None:
            steps_threshold = [0.6, 0.7, 0.7]

        self.__min_face_size = min_face_size
        self.__steps_threshold = steps_threshold
        self.__scale_factor = scale_factor

        #config = tf.ConfigProto(log_device_placement=False)
        #config.gpu_options.allow_growth = True

        #weights = np.load(weights_file).item()
        self.__pnet = PNet().to(device)
        self.__pnet.set_weights('mtcnn/weights/pnet.npy')
        #self.__pnet.set_weights(weights['PNet'])

        self.__rnet = RNet().to(device)
        self.__rnet.set_weights('mtcnn/weights/rnet.npy')
        #self.__rnet.set_weights(weights['RNet'])

        self.__onet = ONet().to(device)
        self.__onet.set_weights('mtcnn/weights/onet.npy')
        #self.__onet.set_weights(weights['ONet'])

    @property
    def min_face_size(self):
        return self.__min_face_size

    @min_face_size.setter
    def min_face_size(self, mfc=20):
        try:
            self.__min_face_size = int(mfc)
        except ValueError:
            self.__min_face_size = 20

    def __compute_scale_pyramid(self, m, min_layer):
        scales = []
        factor_count = 0

        while min_layer >= 12:
            scales += [m * np.power(self.__scale_factor, factor_count)]
            min_layer = min_layer * self.__scale_factor
            factor_count += 1

        return scales

    @staticmethod
    def __correct_bboxes(bboxes, width, height):
        """Crop boxes that are too big and get coordinates
        with respect to cutouts.

        Arguments:
            bboxes: a float numpy array of shape [n, 5],
                where each row is (xmin, ymin, xmax, ymax, score).
            width: a float number.
            height: a float number.

        Returns:
            dy, dx, edy, edx: a int numpy arrays of shape [n],
                coordinates of the boxes with respect to the cutouts.
            y, x, ey, ex: a int numpy arrays of shape [n],
                corrected ymin, xmin, ymax, xmax.
            h, w: a int numpy arrays of shape [n],
                just heights and widths of boxes.

            in the following order:
                [dy, edy, dx, edx, y, ey, x, ex, w, h].
        """

        x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
        w, h = x2 - x1 + 1.0,  y2 - y1 + 1.0
        num_boxes = bboxes.shape[0]

        # 'e' stands for end
        # (x, y) -> (ex, ey)
        x, y, ex, ey = x1, y1, x2, y2

        # we need to cut out a box from the image.
        # (x, y, ex, ey) are corrected coordinates of the box
        # in the image.
        # (dx, dy, edx, edy) are coordinates of the box in the cutout
        # from the image.
        dx, dy = np.zeros((num_boxes,)), np.zeros((num_boxes,))
        edx, edy = w.copy() - 1.0, h.copy() - 1.0

        # if box's bottom right corner is too far right
        ind = np.where(ex > width - 1.0)[0]
        edx[ind] = w[ind] + width - 2.0 - ex[ind]
        ex[ind] = width - 1.0

        # if box's bottom right corner is too low
        ind = np.where(ey > height - 1.0)[0]
        edy[ind] = h[ind] + height - 2.0 - ey[ind]
        ey[ind] = height - 1.0

        # if box's top left corner is too far left
        ind = np.where(x < 0.0)[0]
        dx[ind] = 0.0 - x[ind]
        x[ind] = 0.0

        # if box's top left corner is too high
        ind = np.where(y < 0.0)[0]
        dy[ind] = 0.0 - y[ind]
        y[ind] = 0.0

        return_list = [dy, edy, dx, edx, y, ey, x, ex, w, h]
        return_list = [i.astype('int32') for i in return_list]

        return return_list

    #@staticmethod
    def __get_image_boxes(self,bboxes, image, size=24):
        num_boxes = len(bboxes)
        height, width, _ = image.shape

        [dy, edy, dx, edx, y, ey, x, ex, w, h] = self.__correct_bboxes(bboxes, width, height)
        img_array = np.asarray(image, 'uint8')
        img_boxes = np.zeros((num_boxes, 3, size, size), 'float32')

        for i in range(num_boxes):
            img_box = np.zeros((h[i], w[i], 3), 'uint8')
            img_box[dy[i]:(edy[i] + 1), dx[i]:(edx[i] + 1), :] = img_array[y[i]:(ey[i] + 1), x[i]:(ex[i] + 1), :]
            img_box = cv2.resize(img_box, (size, size), interpolation=cv2.INTER_AREA)
            img_box = img_box.transpose((2, 0, 1))
            img_box = np.expand_dims(img_box, 0)
            img_boxes[i, :, :, :] = np.asarray(img_box, 'float32')

        img_boxes = (img_boxes - 127.5) * 0.0078125
        return img_boxes

    @staticmethod
    def __scale_image(image, scale: float):
        """
        Scales the image to a given scale.
        :param image:
        :param scale:
        :return:
        """
        height, width, _ = image.shape

        width_scaled = int(np.ceil(width * scale))
        height_scaled = int(np.ceil(height * scale))

        im_data = cv2.resize(image, (width_scaled, height_scaled), interpolation=cv2.INTER_AREA)

        # Normalize the image's pixels
        im_data_normalized = (im_data - 127.5) * 0.0078125

        return im_data_normalized

    @staticmethod
    def __generate_bounding_box(probs, offsets, scale, threshold):
        # Generate bounding boxes at places
        # where there is probably a face.
        stride = 2
        cell_size = 12

        # Indices of boxes where there is probably a face
        inds = np.where(probs > threshold)
        if inds[0].size == 0:
            return np.array([])

        # Transformations of bounding boxes
        tx1, ty1, tx2, ty2 = [offsets[0, i, inds[0], inds[1]] for i in range(4)]
        # they are defined as:
        # w = x2 - x1 + 1
        # h = y2 - y1 + 1
        # x1_true = x1 + tx1*w
        # x2_true = x2 + tx2*w
        # y1_true = y1 + ty1*h
        # y2_true = y2 + ty2*h

        offsets = np.array([tx1, ty1, tx2, ty2])
        score = probs[inds[0], inds[1]]

        # To rescale bounding boxes back
        bboxes = np.vstack([
            np.round((stride*inds[1] + 1.0)/scale),
            np.round((stride*inds[0] + 1.0)/scale),
            np.round((stride*inds[1] + 1.0 + cell_size)/scale),
            np.round((stride*inds[0] + 1.0 + cell_size)/scale),
            score, offsets
        ])
        return bboxes.T

    @staticmethod
    def __nms(boxes, threshold, method):
        """
        Non Maximum Suppression.
        :param boxes: np array with bounding boxes.
        :param threshold:
        :param method: NMS method to apply. Available values ('Min', 'Union')
        :return:
        """
        if boxes.size == 0:
            return np.empty((0, 3))

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        s = boxes[:, 4]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        sorted_s = np.argsort(s)

        pick = np.zeros_like(s, dtype=np.int16)
        counter = 0
        while sorted_s.size > 0:
            i = sorted_s[-1]
            pick[counter] = i
            counter += 1
            idx = sorted_s[0:-1]

            xx1 = np.maximum(x1[i], x1[idx])
            yy1 = np.maximum(y1[i], y1[idx])
            xx2 = np.minimum(x2[i], x2[idx])
            yy2 = np.minimum(y2[i], y2[idx])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)

            inter = w * h

            if method is 'Min':
                o = inter / np.minimum(area[i], area[idx])
            else:
                o = inter / (area[i] + area[idx] - inter)

            sorted_s = sorted_s[np.where(o <= threshold)]

        pick = pick[0:counter]

        return pick

    @staticmethod
    def __pad(total_boxes, w, h):
        # compute the padding coordinates (pad the bounding boxes to square)
        tmpw = (total_boxes[:, 2] - total_boxes[:, 0] + 1).astype(np.int32)
        tmph = (total_boxes[:, 3] - total_boxes[:, 1] + 1).astype(np.int32)
        numbox = total_boxes.shape[0]

        dx = np.ones(numbox, dtype=np.int32)
        dy = np.ones(numbox, dtype=np.int32)
        edx = tmpw.copy().astype(np.int32)
        edy = tmph.copy().astype(np.int32)

        x = total_boxes[:, 0].copy().astype(np.int32)
        y = total_boxes[:, 1].copy().astype(np.int32)
        ex = total_boxes[:, 2].copy().astype(np.int32)
        ey = total_boxes[:, 3].copy().astype(np.int32)

        tmp = np.where(ex > w)
        edx.flat[tmp] = np.expand_dims(-ex[tmp] + w + tmpw[tmp], 1)
        ex[tmp] = w

        tmp = np.where(ey > h)
        edy.flat[tmp] = np.expand_dims(-ey[tmp] + h + tmph[tmp], 1)
        ey[tmp] = h

        tmp = np.where(x < 1)
        dx.flat[tmp] = np.expand_dims(2 - x[tmp], 1)
        x[tmp] = 1

        tmp = np.where(y < 1)
        dy.flat[tmp] = np.expand_dims(2 - y[tmp], 1)
        y[tmp] = 1

        return dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph

    @staticmethod
    def __rerec(bbox):
        # convert bbox to square
        h = bbox[:, 3] - bbox[:, 1]
        w = bbox[:, 2] - bbox[:, 0]
        l = np.maximum(w, h)
        bbox[:, 0] = bbox[:, 0] + w * 0.5 - l * 0.5
        bbox[:, 1] = bbox[:, 1] + h * 0.5 - l * 0.5
        bbox[:, 2:4] = bbox[:, 0:2] + np.transpose(np.tile(l, (2, 1)))
        return bbox

    @staticmethod
    def __bbreg(bboxes, offsets):
        # calibrate bounding boxes
        if offsets.shape[1] == 1:
            reg = np.reshape(reg, (offsets.shape[2], offsets.shape[3]))

        w  = bboxes[:, 2] - bboxes[:, 0] + 1
        h  = bboxes[:, 3] - bboxes[:, 1] + 1
        b1 = bboxes[:, 0] + offsets[:, 0] * w
        b2 = bboxes[:, 1] + offsets[:, 1] * h
        b3 = bboxes[:, 2] + offsets[:, 2] * w
        b4 = bboxes[:, 3] + offsets[:, 3] * h
        bboxes[:, 0:4] = np.transpose(np.vstack([b1, b2, b3, b4]))
        return bboxes

    def detect_faces(self, img) -> list:
        """
        Detects bounding boxes from the specified image.
        :param img: image to process
        :return: list containing all the bounding boxes detected with their keypoints.
        """
        if img is None or not hasattr(img, "shape"):
            raise InvalidImage("Image not valid.")

        height, width, _ = img.shape
        stage_status = StageStatus(width=width, height=height)

        m = 12 / self.__min_face_size
        min_layer = np.amin([height, width]) * m

        scales = self.__compute_scale_pyramid(m, min_layer)

        stages = [self.__stage1, self.__stage2, self.__stage3]
        result = [scales, stage_status]

        # We pipe here each of the stages
        for stage in stages:
            result = stage(img, result[0], result[1])

        [total_boxes, points] = result
        bboxes = []

        for bounding_box, keypoints in zip(total_boxes, points):

            bboxes.append({
                    'box': [int(bounding_box[0]), int(bounding_box[1]),
                            int(bounding_box[2]-bounding_box[0]), int(bounding_box[3]-bounding_box[1])],
                    'confidence': bounding_box[-1],
                    'keypoints': {
                        'left_eye': (int(keypoints[0]), int(keypoints[5])),
                        'right_eye': (int(keypoints[1]), int(keypoints[6])),
                        'nose': (int(keypoints[2]), int(keypoints[7])),
                        'mouth_left': (int(keypoints[3]), int(keypoints[8])),
                        'mouth_right': (int(keypoints[4]), int(keypoints[9])),
                    }
                }
            )

        return bboxes

    def __stage1(self, image, scales: list, stage_status: StageStatus):
        """
        First stage of the MTCNN.
        :param image:
        :param scales:
        :param stage_status:
        :return:
        """
        bboxes = np.empty((0, 9))
        status = stage_status

        for scale in scales:
            scaled_image = self.__scale_image(image, scale)
            img = np.asarray(scaled_image, 'float32')
            img = img.transpose((2, 0, 1))
            img = np.expand_dims(img, 0)

            img = Variable(torch.FloatTensor(img), volatile=True)
            out = self.__pnet(img)
            probs = out[1].data.numpy()[0, 1, :, :]
            offsets = out[0].data.numpy()
            # probs: probability of a face at each sliding window
            # offsets: transformations to true bounding boxes

            #out0 = np.transpose(out[0].data.numpy(), (0, 2, 1, 3))
            #out1 = np.transpose(out[1].data.numpy(), (0, 2, 1, 3))

            boxes = self.__generate_bounding_box(probs, offsets, scale, self.__steps_threshold[0])

            # inter-scale nms
            pick = self.__nms(boxes.copy(), 0.5, 'Union')
            if boxes.size > 0 and pick.size > 0:
                boxes = boxes[pick, :]
                bboxes = np.append(bboxes, boxes, axis=0)

        numboxes = bboxes.shape[0]

        if numboxes > 0:
            pick = self.__nms(bboxes.copy(), 0.7, 'Union')
            bboxes = bboxes[pick, :]

            regw = bboxes[:, 2] - bboxes[:, 0]
            regh = bboxes[:, 3] - bboxes[:, 1]

            qq1 = bboxes[:, 0] + bboxes[:, 5] * regw
            qq2 = bboxes[:, 1] + bboxes[:, 6] * regh
            qq3 = bboxes[:, 2] + bboxes[:, 7] * regw
            qq4 = bboxes[:, 3] + bboxes[:, 8] * regh

            bboxes = np.transpose(np.vstack([qq1, qq2, qq3, qq4, bboxes[:, 4]]))
            bboxes = self.__rerec(bboxes.copy())

            bboxes[:, 0:4] = np.fix(bboxes[:, 0:4]).astype(np.int32)
            status = StageStatus(self.__pad(bboxes.copy(), stage_status.width, stage_status.height),
                                 width=stage_status.width, height=stage_status.height)

        return bboxes, status

    def __stage2(self, image, bboxes, stage_status:StageStatus):
        """
        Second stage of the MTCNN.
        :param img:
        :param total_boxes:
        :param stage_status:
        :return:
        """
        num_boxes = bboxes.shape[0]
        if num_boxes == 0:
            return bboxes, stage_status

        img_boxes = self.__get_image_boxes(bboxes, image, size=24)
        img_boxes = Variable(torch.FloatTensor(img_boxes), volatile=True)
        out = self.__rnet(img_boxes)
        offsets = out[0].data.numpy()   # shape [n_boxes, 4]
        probs = out[1].data.numpy()     # shape [n_boxes, 2]

        #out0 = np.transpose(out[0])
        #out1 = np.transpose(out[1])
        keep = np.where(probs[:, 1] > self.__steps_threshold[1])[0]
        bboxes = bboxes[keep]
        bboxes[:, 4] = probs[keep, 1].reshape((-1,))
        offsets = offsets[keep]

        if bboxes.shape[0] > 0:
            pick = self.__nms(bboxes, 0.7, 'Union')
            bboxes = bboxes[pick]
            bboxes = self.__bbreg(bboxes, offsets[pick])
            bboxes = self.__rerec(bboxes.copy())

        return bboxes, stage_status

    def __stage3(self, image, bboxes, stage_status: StageStatus):
        """
        Third stage of the MTCNN.
        :param img:
        :param total_boxes:
        :param stage_status:
        :return:
        """
        num_boxes = bboxes.shape[0]
        if num_boxes == 0:
            return bboxes, np.empty(shape=(0,))

        bboxes = np.fix(bboxes).astype(np.int32)
        status = StageStatus(self.__pad(bboxes.copy(), stage_status.width, stage_status.height),
                             width=stage_status.width, height=stage_status.height)

        img_boxes = self.__get_image_boxes(bboxes, image, size=48)
        if len(img_boxes) == 0:
            return np.empty(shape=(0,)), np.empty(shape=(0,))

        img_boxes = Variable(torch.FloatTensor(img_boxes), volatile=True)
        out = self.__onet(img_boxes)
        landmarks = out[0].data.numpy()     # shape [n_boxes, 10]
        offsets = out[1].data.numpy()       # shape [n_boxes, 4]
        probs = out[2].data.numpy()         # shape [n_boxes, 2]

        keep = np.where(probs[:, 1] > self.__steps_threshold[2])[0]
        bboxes = bboxes[keep]
        bboxes[:, 4] = probs[keep, 1].reshape((-1,))
        offsets = offsets[keep]
        landmarks = landmarks[keep]

        # compute landmark points
        width = bboxes[:, 2] - bboxes[:, 0] + 1.0
        height = bboxes[:, 3] - bboxes[:, 1] + 1.0
        xmin, ymin = bboxes[:, 0], bboxes[:, 1]
        landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1)*landmarks[:, 0:5]
        landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1)*landmarks[:, 5:10]

        if bboxes.shape[0] > 0:
            bboxes = self.__bbreg(bboxes, offsets)
            pick = self.__nms(bboxes, 0.7, 'Min')
            bboxes = bboxes[pick]
            landmarks = landmarks[pick]

        return bboxes, landmarks
