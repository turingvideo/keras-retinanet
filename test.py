import os
import sys
from typing import Dict, Any, List



try:
    from keras_retinanet.bin.train import main as train
except Exception as e:
    print(e)
#
# # TODO: Error logging
# # TODO: Success logging
#
current_path = os.getcwd()

print(current_path)
print(os.listdir(current_path))

# p = os.path.join(current_path, "keras_retinanet/keras_retinanet/utils")
# print(os.listdir(p))


# #
#
# # required hyper params
# # batch_size
# # steps
# # epochs
# # training_catalog_filename
# # validation_catalog_filename
# # weights_file_name
#
# def main():
#     print("###############################################")
#     print("Environments:")
#     for k, v in os.environ.items():
#         print(f"{k}: {v}")
#     print("###############################################")
#
#     print("###############################################")
#     print("argv")
#     for arg in sys.argv:
#         print(arg)
#     print("###############################################")
#
#     hyper_params: Dict[str, Any] = os.environ['SM_HPS']
#     batch_size: int = hyper_params['batch-size']
#     steps: int = hyper_params['steps']
#     epochs: int = hyper_params['epochs']
#     training_catalog_filename: str = hyper_params['training_catalog_filename']
#     validation_catalog_filename: str = hyper_params['validation_catalog_filename']
#     weights_filename: str = hyper_params['weights_file_name']
#
#     weights_folder_path: str = os.environ['SM_CHANNEL_WEIGHTS']
#     training_folder_path: str = os.environ['SM_CHANNEL_TRAINING']
#     validation_folder_path: str = os.environ['SM_CHANNEL_VALIDATION']
#     gpu_number: int = os.environ['SM_NUM_GPUS']
#     tensorboard_folder_path: str = os.environ['SM_OUTPUT_INTERMEDIATE_DIR']
#     model_output_folder_path: str = os.environ['SM_MODEL_DIR']
#
#     weights_file_path: str = os.path.join(weights_folder_path, weights_filename)
#     training_catalog_filename: str = os.path.join(training_folder_path, training_catalog_filename)
#     validation_catalog_file_path: str = os.path.join(validation_folder_path, validation_catalog_filename)
#
#     params: List[str] = [
#         "--freeze-backbone",
#         "--random-transform",
#         "--weights", weights_file_path,
#         "--batch-size", str(batch_size),
#         "--steps", str(steps),
#         "--epochs", str(epochs),
#         "--multi-gpu", str(gpu_number),
#         "--tensorboard-dir", tensorboard_folder_path,
#         "--snapshot-path", model_output_folder_path
#     ]
#
#     if gpu_number > 1:
#         params += ["--multi-gpu-force"]
#
#     params += [
#         "via",
#         training_catalog_filename,
#         validation_catalog_file_path
#     ]
#     train(params)
#
# # print("---------------------------")
#
# # batch-size
# # weights
# # steps
# # epochs
# # gpu, double check if this is used
# # multi-gpu
# # tensorboard-dir
# # snapshot-path
# # train-catalog
# # validate-catalog
#
# # output_data_dir = os.environ['SM_OUTPUT_DATA_DIR']
# # model_dir = os.environ['SM_MODEL_DIR']
# # output_intermediate_dir = os.environ['SM_OUTPUT_INTERMEDIATE_DIR']
# # train_dir = os.environ['SM_CHANNEL_TRAINING']
# # pretrain_dir = os.environ['SM_CHANNEL_PRETRAIN']
#
# # if not os.path.exists(output_intermediate_dir):
# #     os.mkdir(output_intermediate_dir)
# #
# # print(os.listdir(train_dir))
#
# # with open(os.path.join(output_data_dir, "log1.txt"), 'w') as f:
# #     f.write("log1")
# #
# # with open(os.path.join(output_data_dir, "log2.txt"), 'w') as f:
# #     f.write("log2")
# #
# # with open(os.path.join(model_dir, "model1.txt"), 'w') as f:
# #     f.write("model1")
# #
# # with open(os.path.join(model_dir, "model2.txt"), 'w') as f:
# #     f.write("model2")
# #
# # with open(os.path.join(output_intermediate_dir, "inter1.txt"), 'w') as f:
# #     f.write("inter1")
# #
# # with open(os.path.join(output_intermediate_dir, "inter2.txt"), 'w') as f:
# #     f.write("inter2")
#
# # for a in sys.argv:
# #     print(a)
#
# # weights_path = os.path.join(train_dir, "resnet50_coco_best_v2.1.0.h5")
# # train_catalog_path = os.path.join(train_dir, "train_catalog")
# # val_catalog_path = os.path.join(train_dir, "val_catalog")
#
# # if __name__ == '__main__':
# #     main(["--freeze-backbone",
# #           "--random-transform",
# #           "--weights", weights_path,
# #           "--batch-size", "8",
# #           "--steps", "10",
# #           "--gpu", "0,1",
# #           "--multi-gpu", "2",
# #           "--multi-gpu-force",
# #           "--epochs", "20",
# #           "--tensorboard-dir", output_intermediate_dir,
# #           "via",
# #           train_catalog_path,
# #           val_catalog_path])