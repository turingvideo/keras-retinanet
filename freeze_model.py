import argparse
import os

import keras
import tensorflow.compat.v1 as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from keras_retinanet import models


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


def adapt_detection_model(frozen_graph, input_name_maps, output_name_maps):
    graph = tf.Graph()
    with graph.as_default():
        input_map = {}
        for name, node_info in input_name_maps.items():
            node_shape = node_info['shape']
            new_name = node_info['new_name']
            new_input_tensor = tf.placeholder(
                tf.float32, node_shape, name=new_name)
            input_map[name + ":0"] = new_input_tensor

        tf.import_graph_def(frozen_graph, name='', input_map=input_map)

        for name, new_name in output_name_maps.items():
            tensor = graph.get_tensor_by_name(name + ":0")
            if new_name == 'detection_boxes':
                image_tensor = graph.get_tensor_by_name('image_tensor:0')
                height = tf.cast(tf.shape(image_tensor)[1], tf.float32)
                width = tf.cast(tf.shape(image_tensor)[2], tf.float32)
                shape_factor = tf.stack([width, height, width, height])
                ratio_boxes = tensor / shape_factor
                reordered_boxes = tf.concat([ratio_boxes[:, :, 1:2],
                                             ratio_boxes[:, :, 0:1],
                                             ratio_boxes[:, :, 3:4],
                                             ratio_boxes[:, :, 2:3]],
                                            axis=2, name=new_name)

            elif new_name == 'detection_classes':
                classes = tf.add(tensor, 1, name=new_name)

            else:
                new_tensor = tf.identity(tensor, name=new_name)

    return graph


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True,
                        help="The input keras model file name.")
    parser.add_argument('-b', '--backbone', default='resnet50',
                        help="The backbone model name.")
    parser.add_argument('-o', '--output', required=True,
                        help="The output frozen model file name")
    args = parser.parse_args()

    input_model_file = args.input
    output_model_file = args.output
    backbone=args.backbone

    keras.backend.set_learning_phase(0)
    model = models.load_model(input_model_file, backbone_name=backbone)

    print('Input tensors: {}'.format(model.inputs))
    print('Original output tensors: {}'.format(model.outputs))

    if len(model.inputs) > 1:
        print('The model should have only one input tensor, given {}.'
              .format(len(model.inputs)))
        print('Convert model failed.')
        exit()

    if len(model.outputs) < 3:
        print('The model should have at least three output tensors, '
              'leading by (boxes, scores, classes)')
        print('Convert model failed.')
        exit()

    input_name = model.inputs[0].op.name
    output_names=[out.op.name for out in model.outputs]
    # print(output_names)

    frozen_graph = freeze_session(keras.backend.get_session(),
                                  output_names=output_names)

    input_name_maps = {
        input_name: {
            "shape": model.inputs[0].shape,
            "new_name": "image_tensor",
        }
    }
    output_name_maps = {
        output_names[0]: "detection_boxes",
        output_names[1]: "detection_scores",
        output_names[2]: "detection_classes",
    }

    frozen_graph = adapt_detection_model(frozen_graph, input_name_maps, output_name_maps)

    outdir = os.path.dirname(output_model_file)
    filename = os.path.basename(output_model_file)
    tf.train.write_graph(frozen_graph, outdir, filename, as_text=False)

