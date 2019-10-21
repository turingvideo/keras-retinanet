# from ..preprocessing.generator import Generator
import copy
import json
import numpy as np
import os

from keras_retinanet.preprocessing.generator import Generator
from keras_retinanet.utils.image import read_image_bgr


class ViaGenerator(Generator):

    def __init__(self, via_catalog_file_path, **kwargs):
        if not os.path.exists(via_catalog_file_path):
            raise IOError("File not exists: {}".format(via_catalog_file_path))

        self.categories = {0: "person"}
        self.labels = {"person": 0}

        # if via_file_path.endswith('.json'):
        #     self.image_paths, self.annotations = \
        #         self.load_via_annotation(via_file_path)
        # else:
        with open(via_catalog_file_path, 'r') as f:
            via_catalog_folder_path = os.path.dirname(via_catalog_file_path)
            catalog_content = json.load(via_catalog_file_path)
            via_folder_path = catalog_content["via_folder_path"]
            relative_via_paths = catalog_content["relative_via_paths"]
            via_file_paths = [os.path.join(via_folder_path, relative_via_path)
                              for relative_via_path
                              in relative_via_paths]
        print(via_file_paths)
        self.image_paths = []
        self.annotations = []
        for via_file_path in via_file_paths:
            try:
                image_paths, annotations = \
                    self.load_via_annotation(via_file_path)
                self.image_paths += image_paths
                self.annotations += annotations

            except IOError as e:
                print('Failed to parse via annotation file: {}, {}'
                      .format(via_file_path, e))

        super(ViaGenerator, self).__init__(**kwargs)

        # for i in range(self.size()):
        #     try:
        #         self.load_image(i)
        #     except Exception as e:
        #         print('Failed to read im {}'.format(self.image_list[i]))
        #     if i > 0 and i % 100 == 0:
        #         print('checked {} images'.format(i))
        # print('done loading via annotation')

    def load_via_annotation(self, via_file_path):

        via_file_folder_path = os.path.dirname(via_file_path)

        with open(via_file_path, 'r') as f:
            raw_via_annotations = json.load(f)

        raw_image_folder_path = raw_via_annotations['_via_settings']['core']['default_filepath']

        if os.path.isabs(raw_image_folder_path):
            image_folder_path = raw_image_folder_path
        else:
            image_folder_path = os.path.join(via_file_folder_path, raw_image_folder_path)

        via_annotations = raw_via_annotations['_via_img_metadata']
        image_list, annotations = self.via_to_bbox(via_annotations)

        image_paths = []
        for image_name in image_list:
            image_path = os.path.join(image_folder_path, image_name)
            image_paths.append(image_path)

        return image_paths, annotations

    def via_to_bbox(self, via_annotations):
        image_filenames = []
        annotations = []
        for key, via_anno in via_annotations.items():
            bboxes = []
            labels = []
            for region in via_anno['regions']:
                category = region['region_attributes']['name']
                shape_attributes = region['shape_attributes']
                bbox = [
                    shape_attributes['x'],
                    shape_attributes['y'],
                    shape_attributes['x'] + shape_attributes['width'],
                    shape_attributes['y'] + shape_attributes['height'],
                ]
                if category in self.labels:
                    labels.append(self.labels[category])
                    bboxes.append(bbox)
                else:
                    print('unknown category {}'.format(category))

            image_filenames.append(via_anno['filename'])
            if len(labels) == 0:
                labels = np.empty((0,), dtype='int32')
            else:
                labels = np.asarray(labels, dtype='int32')

            if len(bboxes) == 0:
                bboxes = np.empty((0, 4), dtype='float64')
            else:
                bboxes = np.asarray(bboxes, dtype='float64')
            annotations.append(dict(labels=labels, bboxes=bboxes))

        return image_filenames, annotations

    def size(self):
        return len(self.annotations)

    def num_classes(self):
        return len(self.categories)

    def has_label(self, label):
        return label in self.categories

    def has_name(self, name):
        return name in self.labels

    def name_to_label(self, name):
        return self.labels[name]

    def label_to_name(self, label):
        return self.categories[label]

    def image_aspect_ratio(self, image_index):
        return 1

    def load_image(self, image_index):
        path = self.image_paths[image_index]
        return read_image_bgr(path)

    def load_annotations(self, image_index):
        return copy.deepcopy(self.annotations[image_index])


if __name__ == "__main__":
    viagen = ViaGenerator(
        '/media/fwang/Data1/PedestrianDataset/WIDER Person Challenge 2019/Annotations/val_via_no_filter.json')
    # viagen = ViaGenerator('./data/test_via_files.txt')
    print(viagen.size())
    image_index = 0
    im = viagen.load_image(image_index)
    print(im.shape)
    anno = viagen.load_annotations(image_index)
    print(anno)
