from keras.backend import cast, exp, reshape, tile, sigmoid
from keras.engine.training import concat
from keras.layers import Conv2D, Input, BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D
from keras.layers.merge import add, concatenate
from keras.models import Model
import tensorflow as tf
import cv2


"""Parsira konfiguracioni fajl neuronske mreze"""
def parse_cfg(cfg_file_path):
    with open(cfg_file_path, 'r') as file:
        lines = [line.rstrip('\n') for line in file if line != '\n' and line[0] != '#'] # preskacemo prazne redove i komentare

        layer_config = {}
        layers = []
        for line in lines:
            # Ako je linija definicija novog bloka
            if line[0] == '[':
                line = 'type=' + line[1:-1].rstrip()
                if layer_config:
                    layers.append(layer_config)
                    layer_config = {}

            key, value = line.split('=')
            layer_config[key.rstrip()] = value.lstrip()

        layers.append(layer_config)
        return layers


def YOLOv3Net(cfg_file_path, model_size, num_classes):
    layers = parse_cfg(cfg_file_path)

    outputs = {}
    output_filters = []
    filters = []
    out_pred = []
    scale = 0

    inputs = input_image = Input(shape=model_size)
    inputs /= 255.0

    for i, layer in enumerate(layers[1:]):

        if layer['type'] == 'convolutional':
            activation = layer['activation']
            filters = int(layer['filters'])
            kernel_size = int(layer['size'])
            stride = int(layer['stride'])

            if stride > 1:
                inputs = ZeroPadding2D(((1, 0), (1, 0)))(inputs)

            inputs = Conv2D(filters,
                            kernel_size,
                            strides=stride,
                            padding='valid' if stride > 1 else 'same', 
                            name='conv_' + str(i),
                            use_bias=False if 'batch_normalize' in layer else True)(inputs)

            if 'batch_normalize' in layer:
                inputs = BatchNormalization(name='bnorm_' + str(i))(inputs)
                inputs = LeakyReLU(alpha=0.1, name='leaky_' + str(i))(inputs)
        

        elif layer['type'] == 'upsample':
            stride = int(layer['stride'])
            inputs = UpSampling2D(stride)(inputs)

        
        elif layer['type'] == 'route':
            layer['layers'] = layer['layers'].split(',')
            print(f"LAYER: {layer['layers']}")
            start = int(layer['layers'][0])

            if len(layer['layers']) > 1:
                end = int(layer['layers'][1]) - i
                filters = output_filters[i+start] + output_filters[end]  # ili [i+end] samo, testiracemo
                inputs = concat([outputs[i+start], outputs[i+end]], axis=-1)
                print(f'INDICES: \n-start={start}, end={end},\n-[i+start]={i+start}, [i+end]={i+end}')
            else:
                filters = output_filters[i+start]
                inputs = outputs[i+start]

        
        elif layer['type'] == 'shortcut':
            from_ = int(layer['from'])
            inputs = outputs[i-1] + outputs[i+from_]

        
        elif layer['type'] == 'yolo':
            mask = layer['mask'].split(",")
            mask = [int(x) for x in mask]
            anchors = layer['anchors'].split(',')
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]
            n_anchors = len(anchors)

            out_shape = inputs.get_shape().as_list()

            inputs = reshape(inputs, [-1, n_anchors * out_shape[1] * out_shape[2], 5 + num_classes])

            box_centers = inputs[:, :, 0:2]
            box_shapes = inputs[:, :, 2:4]
            confidence = inputs[:, :, 4:5]
            classes = inputs[:, :, 5:num_classes + 5]

            box_centers = sigmoid(box_centers)
            confidence = sigmoid(confidence)
            classes = sigmoid(classes)

            anchors = tile(anchors, [out_shape[1] * out_shape[2], 1])
            box_shapes = exp(box_shapes) * cast(anchors, dtype=tf.float32)

            x = tf.range(out_shape[1], dtype=tf.float32)
            y = tf.range(out_shape[2], dtype=tf.float32)
            
            cx, cy = tf.meshgrid(x, y)
            cx = reshape(cx, (-1, 1))
            cy = reshape(cy, (-1, 1))
            cxy = concat([cx, cy], axis=-1)
            cxy = tile(cxy, [1, n_anchors])
            cxy = reshape(cxy, [1, -1, 2])
            strides = (input_image.shape[1] // out_shape[1], \
                        input_image.shape[2] // out_shape[2])
            box_centers = (box_centers + cxy) * strides

            prediction = concat([box_centers, box_shapes, confidence, classes], axis=-1)

            if scale:
                out_pred = concat([out_pred, prediction], axis=1)
            else:
                out_pred = prediction
                scale = 1

        outputs[i] = inputs
        output_filters.append(filters)


    model = Model(input_image, out_pred)
    model.summary()
    return model
        

            
