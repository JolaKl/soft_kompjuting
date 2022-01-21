import numpy as np
from yolo3 import YOLOv3Net
from yolo3 import parse_cfg

def load_weights(model, cfg_file, weight_file):
    fp = open(weight_file, 'rb')
    np.fromfile(fp, dtype=np.int32, count=5)

    blocks = parse_cfg(cfg_file)

    for i, block in enumerate(blocks[1:]):
        if block['type'] == 'convolutional':
            conv_layer = model.get_layer('conv_' + str(i))
            print("layer: ",i+1,conv_layer)

            filters = conv_layer.filters
            k_size = conv_layer.kernel_size[0]
            in_dim = conv_layer.input_shape[-1]

            if 'batch_normalize' in block:
                norm_layer = model.get_layer('bnorm_' + str(i))
                print("layer: ",i+1,norm_layer)
                size = np.prod(norm_layer.get_weights()[0].shape)

                bn_weights = np.fromfile(fp, dtype=np.float32, count = 4*filters)
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

            else:
                conv_bias = np.fromfile(fp, dtype=np.float32, count=filters)

            
            conv_shape = (filters, in_dim, k_size, k_size)
            conv_weights = np.fromfile(
                fp, dtype=np.float32, count=np.product(conv_shape)
            )

            conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

            if 'batch_normalize' in block:
                norm_layer.set_weights(bn_weights)
                conv_layer.set_weights([conv_weights])
            else:
                conv_layer.set_weights([conv_weights, conv_bias])

    assert len(fp.read()) == 0, 'failed to read all data'
    fp.close()


def main(save_weights=False):
    weight_file = '../yolov3_training_last.weights'
    cfg_file = '../yolov3_testing.cfg'

    model_size = (416, 416, 3)
    num_classes = 2

    model = YOLOv3Net(cfg_file, model_size, num_classes)
    load_weights(model, cfg_file, weight_file)

    print('Loaded model weights')

    try:
        if save_weights:
            model.save_weights('weights/yolov3_weights.tf')
            print('\nThe file \'yolov3_weights.tf\' has been saved successfully.')
    except IOError:
        print("Couldn't write the file \'yolov3_weights.tf\'.")

if __name__ == '__main__':
    main(True)

