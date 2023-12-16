import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
from shapely.wkt import loads as wkt_loads
import tifffile as tiff
import os
import random
from keras import backend as keras_backend
from keras.models import Model
from keras.layers import Input, Convolution2D, Convolution2DTranspose, MaxPooling2D, Dropout, concatenate
from keras.optimizers import legacy, schedules
from keras.callbacks import TensorBoard
import gc

# these are the classes defined in the data file
ORIGINAL_CLASS_LIST = [
    "Buildings", "Misc Manmade structures", "Road",
    "Track", "Trees", "Crops", "Waterway", "Standing water",
    "Vehicle Large", "Vehicle Small"
]

# these are the classes we will actually use
CLASS_LIST = [
    "Buildings", "Misc Manmade structures",
    "Road", "Track",
    "Trees", "Crops",
    "Water",
    "Background"
]

# average of occurence for each class
AVERAGE_CLASS_FREQUENCIES = {
    "Buildings": 0.03,
    "Misc Manmade structures": 0.009,
    "Road": 0.008,
    "Track": 0.03,
    "Trees": 0.1,
    "Crops": 0.25,
    "Water": 0.006,
}

CLASS_COLORS = {
    "Buildings": [0x96, 0x90, 0x07, 0],
    "Misc Manmade structures": [0xA8, 0xD1, 0xD1, 0],
    "Road": [0x4B, 0x4B, 0x50, 0],
    "Track":[0x96, 0x4B, 0x00, 0],
    "Trees": [0xAA, 0xDD, 0x2D, 0],
    "Crops":[0xEB, 0xCC, 0x95, 0],
    "Water": [0x24, 0x88, 0xD5, 0],
    "Background": [0xFF, 0xFF, 0xFF, 0]
}

# mapping from original class index to new index
# -1 is used to skip a class
ORIGINAL_INDEX_MAP = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 6,
    8: -1,
    9: -1
}

ORIGINAL_CLASSES = len(ORIGINAL_CLASS_LIST)
CLASSES = len(CLASS_LIST)
CHANNELS = 8
TRAINING_CYCLES = 1
EPOCHS = 10
INPUT_SIZE = 128
BATCH_SIZE = 64
EPSILON = 1e-12

inDir = './'
os.makedirs(inDir + 'kaggle/data', exist_ok=True)
os.makedirs(inDir + 'kaggle/figures/simple', exist_ok=True)
os.makedirs(inDir + 'kaggle/figures/detailed', exist_ok=True)
# os.makedirs(inDir + 'kaggle/msk', exist_ok=True)
os.makedirs(inDir + 'kaggle/weights', exist_ok=True)
os.makedirs(inDir + 'kaggle/logs', exist_ok=True)

# data op
def _convert_coordinates_to_raster(coords, img_size, xymax):
    Xmax, Ymax = xymax
    H, W = img_size
    W1 = 1.0 * W * W / (W + 1)
    H1 = 1.0 * H * H / (H + 1)
    xf = W1 / Xmax
    yf = H1 / Ymax
    coords[:, 1] *= yf
    coords[:, 0] *= xf
    coords_int = np.round(coords).astype(np.int32)

    return coords_int

# data op
def _get_xmax_ymin(grid_sizes_panda, imageId):
    xmax, ymin = grid_sizes_panda[grid_sizes_panda.ImageId == imageId].iloc[0, 1:].astype(float)

    return (xmax, ymin)

# data op
def _get_polygon_list(wkt_list_pandas, imageId, cType):
    df_image = wkt_list_pandas[wkt_list_pandas.ImageId == imageId]
    multipoly_def = df_image[df_image.ClassType == cType].MultipolygonWKT
    polygonList = []
    if len(multipoly_def) > 0:
        assert len(multipoly_def) == 1
        polygonList = wkt_loads(multipoly_def.values[0])

    return polygonList

# data op
def _get_and_convert_contours(polygonList, raster_img_size, xymax):
    perim_list = []
    interior_list = []

    if polygonList is None:
        return None
    
    for k in range(len(polygonList.geoms)):
        # print(k)
        poly = polygonList.geoms[k]
        perim = np.array(list(poly.exterior.coords))
        perim_c = _convert_coordinates_to_raster(perim, raster_img_size, xymax)
        perim_list.append(perim_c)
        for pi in poly.interiors:
            interior = np.array(list(pi.coords))
            interior_c = _convert_coordinates_to_raster(interior, raster_img_size, xymax)
            interior_list.append(interior_c)

    return perim_list, interior_list

# data op
def _plot_mask_from_contours(raster_img_size, contours, class_value=1):
    img_mask = np.zeros(raster_img_size, np.uint8)
    if contours is None:
        return img_mask
    perim_list, interior_list = contours
    cv2.fillPoly(img_mask, perim_list, class_value)
    cv2.fillPoly(img_mask, interior_list, 0)

    return img_mask

# data op
def generate_mask_for_image_and_class(raster_size, imageId, class_type, grid_sizes_panda, wkt_list_pandas):
    xymax = _get_xmax_ymin(grid_sizes_panda, imageId)
    polygon_list = _get_polygon_list(wkt_list_pandas, imageId, class_type)
    contours = _get_and_convert_contours(polygon_list, raster_size, xymax)
    mask = _plot_mask_from_contours(raster_size, contours, 1)

    return mask

# data op
def mask_for_polygons(polygons, im_size):
    img_mask = np.zeros(im_size, np.uint8)
    if not polygons:
        return img_mask
    
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons.geoms]
    interiors = [int_coords(pi.coords) for poly in polygons.geoms
                 for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)

    gc.collect()

    return img_mask

# data op
def multispectral(image_id):
    filename = os.path.join(inDir, 'sixteen_band', '{}_M.tif'.format(image_id))
    img = tiff.imread(filename)
    img = np.rollaxis(img, 0, 3)

    return img

def stretch_n(bands, lower_percent=2, higher_percent=98):
    out = np.zeros_like(bands).astype(np.float32)
    n = bands.shape[2]

    for i in range(n):
        band = bands[:, :, i]
        a = 0 # minimum value after processing
        b = 1 # maximum value after processing
        c = np.percentile(band, lower_percent)
        d = np.percentile(band, higher_percent)
        t = a + (band - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[:, :, i] = t

    return out

# neural op
def jaccard_coef(y_true, y_pred):
    intersection = keras_backend.sum(y_true * y_pred, axis=[0,1,2])
    sum_ = keras_backend.sum(y_true + y_pred, axis=[0,1,2])
    jac = (intersection + EPSILON) / (sum_ - intersection + EPSILON)

    return keras_backend.mean(jac)

# neural op
def jaccard_coef_int(y_true, y_pred):
    y_pred_pos = keras_backend.round(keras_backend.clip(y_pred, 0, 1))
    intersection = keras_backend.sum(y_true * y_pred_pos, axis=[0,1,2])
    sum_ = keras_backend.sum(y_true + y_pred_pos, axis=[0,1,2])
    jac = (intersection + EPSILON) / (sum_ - intersection + EPSILON)

    return keras_backend.mean(jac)

# neural op
def jaccard_loss(y_true, y_pred):
    return -jaccard_coef(y_true, y_pred)

# neural op
def prepare_training_data():
    print("preparing training data")

    # if necessary files already exist, skip this step
    if os.path.isfile(inDir + '/kaggle/data/input_training_%d.npy' % CLASSES) and os.path.isfile(inDir + '/kaggle/data/output_training_%d.npy' % CLASSES):
        print("data already prepared, skipping")
        return

    dataFrame = pd.read_csv(inDir + '/train_wkt_v4.csv')
    gridSizes = pd.read_csv(inDir + '/grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)

    ids = sorted(dataFrame.ImageId.unique())
    image_count = len(ids)
    print("image count", image_count, "ids:", ids)

    dim = 835
    input = np.zeros((image_count, dim, dim, CHANNELS))
    output = np.zeros((image_count, dim, dim, CLASSES-1))

    for i in range(image_count):
        id = ids[i]
        img = multispectral(id)
        img = stretch_n(img)

        # handle input
        input[i] = img[:dim, :dim, :]
        
        # handle output
        for z_org in range(ORIGINAL_CLASSES):
            # remap original index
            z = ORIGINAL_INDEX_MAP[z_org]
            if z == -1:
                continue 
            
            output[i, :, :, z] = generate_mask_for_image_and_class((img.shape[0], img.shape[1]), id, z_org + 1, gridSizes, dataFrame)[:dim, :dim]

    # handle background after everything else
    exists_mask = np.sum(output, axis=3)
    background_mask = (exists_mask == 0).astype(np.float32)
    background_mask = np.reshape(background_mask, (background_mask.shape[0], background_mask.shape[1], background_mask.shape[2], 1))
    output = np.concatenate((output, background_mask), axis=3)

    print ("input shape", input.shape, "input data mapped to range: ", np.amin(input), np.amax(input))
    print("output shape", output.shape, "output data mapped to range: ", np.amin(output), np.amax(output))

    np.save(inDir + '/kaggle/data/input_training_%d' % CLASSES, input)
    np.save(inDir + '/kaggle/data/output_training_%d' % CLASSES, output)

    gc.collect()

# data op
def get_patches(inputs, outputs, amount):
    input_patches, output_patches = [], []

    while len(input_patches) < amount:
        # rotate images used for patch extraction
        index = len(input_patches) % inputs.shape[0]
        input = inputs[index]
        output = outputs[index]

        # select a starting point
        x_max, y_max = input.shape[0] - INPUT_SIZE, input.shape[1] - INPUT_SIZE
        x_start = random.randint(0, x_max)
        y_start = random.randint(0, y_max)

        # create patch of size (INPUT_SIZE x INPUT_SIZE)
        input_patch = input[x_start:x_start + INPUT_SIZE, y_start:y_start + INPUT_SIZE]
        output_patch = output[x_start:x_start + INPUT_SIZE, y_start:y_start + INPUT_SIZE]

        # skip background here
        for j in range(CLASSES - 1):
            # calculate how many pixels with this class are in the patch
            class_pixels = np.sum(output_patch[:, :, j])
            # calculate ratio of pixels with this class in the patch
            ratio = class_pixels / (INPUT_SIZE ** 2)
            # if the ratio is good enough, use it for validation
            if ratio >= AVERAGE_CLASS_FREQUENCIES[CLASS_LIST[j]] / 3:
                # perform a horizontal flip with probability 0.5
                if random.uniform(0, 1) > 0.5:
                    input_patch = input_patch[::-1]
                    output_patch = output_patch[::-1]
                # perform a vertical flip with probability 0.5
                if random.uniform(0, 1) > 0.5:
                    input_patch = input_patch[:, ::-1]
                    output_patch = output_patch[:, ::-1]

                input_patches.append(input_patch)
                output_patches.append(output_patch)

                # prevent same patch getting triggered by different classes 
                break

    input_patches = 2 * np.array(input_patches) - 1
    output_patches = np.array(output_patches)
    print ('input patches', input_patches.shape, np.amin(input_patches), np.amax(input_patches))
    print ('output patches', output_patches.shape, np.amin(output_patches), np.amax(output_patches))

    gc.collect()

    return input_patches, output_patches

# neural op
def evaluate_jacc(model, img, msk):
    print("evaluate predictions by calculating jaccard score")
    # get some new patches to calculate the jaccard score on new data
    x_val, y_val = get_patches(img, msk, BATCH_SIZE * 30)
    y_pred = model.predict(x_val, batch_size=4)

    print ("prediction shape: ", y_pred.shape, " expected shape: ", y_val.shape)

    scores = []

    for i in range(CLASSES):
        class_true = y_val[:, :, :, i]
        class_pred = y_pred[:, :, :, i]
        class_score = jaccard_coef_int(class_true, class_pred).numpy()
        scores.append(class_score)
        print('class score for', CLASS_LIST[i], class_score)

    score = sum(scores) / CLASSES
    print('average score for all classes', score)

    del(x_val, y_val, y_pred)
    gc.collect()

def double_conv_block(x, n_filters):
    x = Convolution2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
    x = Convolution2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)

    return x

def downsample_block(x, n_filters):
    # Conv2D twice with ReLU activation
    f = double_conv_block(x, n_filters)
    # downsample
    p = MaxPooling2D(2)(f)
    p = Dropout(0.3)(p)

    return f, p

def upsample_block(x, conv_features, n_filters):
    # upsample
    x = Convolution2DTranspose(n_filters, 3, 2, padding="same")(x)
    x = concatenate([x, conv_features])
    x = Dropout(0.3)(x)
    # Conv2D twice with ReLU activation
    x = double_conv_block(x, n_filters)

    return x

# neural op
def get_unet():
    # inputs
    inputs = Input((INPUT_SIZE, INPUT_SIZE, CHANNELS))

    # encoder: contracting path - downsample
    f1, p1 = downsample_block(inputs, 32)
    f2, p2 = downsample_block(p1, 64)
    f3, p3 = downsample_block(p2, 128)
    f4, p4 = downsample_block(p3, 256)
    # bottleneck - stop contracting
    bottleneck = double_conv_block(p4, 512)
    # decoder: expanding path - upsample
    u6 = upsample_block(bottleneck, f4, 256)
    u7 = upsample_block(u6, f3, 128)
    u8 = upsample_block(u7, f2, 64)
    u9 = upsample_block(u8, f1, 32)
    
    # outputs
    outputs = Convolution2D(CLASSES, 1, padding="same", activation = "softmax")(u9)

    # model
    model = Model(inputs=inputs, outputs=outputs)

    # create a learning rate scheduler
    learning_rate_scheduler = schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=150,
        decay_rate=0.90,
        staircase=False
    )

    # compile the model
    model.compile(optimizer=legacy.Adam(learning_rate=learning_rate_scheduler), loss=jaccard_loss, metrics=["accuracy", jaccard_coef_int])

    return model

# neural op
def train_net():
    print ("train net")
    inputs = np.load(inDir + '/kaggle/data/input_training_%d.npy' % CLASSES)
    outputs = np.load(inDir + '/kaggle/data/output_training_%d.npy' % CLASSES)

    model = get_unet()

    # for visualization
    model.save("keras_model.h5")

    # TODO: reenable in the future 
    #model.load_weights('../input/trained-weight/unet_10_jk0.7565')

    for i in range(TRAINING_CYCLES):
        x_trn, y_trn = get_patches(inputs, outputs, BATCH_SIZE * 30)
        # create tensorboard for monitoring
        tensorboard = TensorBoard(log_dir=inDir+'kaggle/logs',update_freq='batch')
        # fit the model to the data
        model.fit(x_trn, y_trn, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, shuffle=True, callbacks=[tensorboard])
        
        del(x_trn, y_trn)
        gc.collect()

        evaluate_jacc(model, inputs, outputs)

    model.save_weights(inDir +'/kaggle/weights/unet_%d' % CLASSES)

def rgb_for_img(img):
    # this creates RGB image from multispectral data
    rgb = np.zeros((img.shape[0], img.shape[1], 3))
    rgb[:,:,0] = img[:,:,4] #red channel
    rgb[:,:,1] = img[:,:,2] #green channel
    rgb[:,:,2] = img[:,:,1] #blue channel

    return rgb

def true_mask_for_img(img, id):
    # this creates true mask for image
    # we want class to be first dimension (makes drawing easier)
    true_class_msk = np.zeros((img.shape[0], img.shape[1], CLASSES-1))
    dataFrame = pd.read_csv(inDir + '/train_wkt_v4.csv')
    gridSizes = pd.read_csv(inDir + '/grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)

    # handle normal classes
    for z_org in range(ORIGINAL_CLASSES):
        # remap original index
        z = ORIGINAL_INDEX_MAP[z_org]
        if z == -1:
            continue

        true_class_msk[:, :, z] = generate_mask_for_image_and_class((img.shape[0], img.shape[1]), id, z_org + 1, gridSizes, dataFrame)

    exists_mask = np.sum(true_class_msk, axis=2)
    background_mask = (exists_mask == 0).astype(np.float32)
    background_mask = np.reshape(background_mask, (background_mask.shape[0], background_mask.shape[1], 1))
    true_mask = np.concatenate((true_class_msk, background_mask), axis=2)

    # we want class to be first dimension (makes drawing easier)
    mask_by_class = true_mask.transpose(2, 0, 1)

    return mask_by_class

def prediction_mask_for_img(img, model):
    # this creates prediction mask for image
    x = stretch_n(img)
    x_scale = round(img.shape[0] / INPUT_SIZE)
    y_scale = round(img.shape[1] / INPUT_SIZE)
    input = np.zeros((INPUT_SIZE * x_scale, INPUT_SIZE * y_scale, CHANNELS)).astype(np.float32)
    output = np.zeros((INPUT_SIZE * x_scale, INPUT_SIZE * y_scale, CLASSES)).astype(np.float32)
    input[:img.shape[0], :img.shape[1], :] = x

    # this processes the input image in INPUT_SIZE*INPUT_SIZE patches
    # since the neural network can only ingest them this way
    for i in range(0, x_scale):
        line = []
        for j in range(0, y_scale):
            line.append(input[i * INPUT_SIZE:(i + 1) * INPUT_SIZE, j * INPUT_SIZE:(j + 1) * INPUT_SIZE])

        x = 2 * np.array(line) - 1
        tmp = model.predict(x, batch_size=4)

        for j in range(tmp.shape[0]):
            output[i * INPUT_SIZE:(i + 1) * INPUT_SIZE, j * INPUT_SIZE:(j + 1) * INPUT_SIZE, :] = tmp[j]

    # we want class to be first dimension (makes drawing easier)
    pred_by_class = output.transpose(2, 0, 1)

    # resize the predictions to match the image size
    return pred_by_class[:, :img.shape[0], :img.shape[1]]

def full_mask(img, mask):
    mask = mask.transpose(1, 2, 0)
    result = np.zeros((img.shape[0], img.shape[1], 4)).astype(np.uint8)

    for x in range(mask.shape[0]):
        for y in range(mask.shape[1]):
            distr = mask[x][y]
            argmax = np.argmax(distr)
            result[x][y] = CLASS_COLORS[CLASS_LIST[argmax]]
            result[x][y][3] = 255 * distr[argmax]
    
    return result

def check_predict_simple(id):
    print("check_predict (simple)")
    model = get_unet()
    model.load_weights(inDir +'/kaggle/weights/unet_%d' % CLASSES)

    m = multispectral(id)
    print("image shape: ", m.shape)

    rgb = rgb_for_img(m)
    true_msk = true_mask_for_img(m, id)
    pred_msk = prediction_mask_for_img(m, model)

    full_msk_true = full_mask(m, true_msk)
    full_msk_pred = full_mask(m, pred_msk)

    # create the plot
    plt.figure(figsize=(20,20))

    # plot the original image
    ax1 = plt.subplot(131)
    ax1.set_title('image ID: ' + id)
    ax1.imshow(stretch_n(rgb))

    # plot image of full mask
    ax4 = plt.subplot(132)
    ax4.set_title("all actual")
    ax4.imshow(full_msk_true)

    # plot image of full mask
    ax4 = plt.subplot(133)
    ax4.set_title("all predicted")
    ax4.imshow(full_msk_pred)

    # save the plot
    plt.savefig(inDir + '/kaggle/figures/simple/' + id + '-All')

def check_predict_detailed(id):
    print("check_predict (detailed)")
    model = get_unet()
    model.load_weights(inDir +'/kaggle/weights/unet_%d' % CLASSES)

    m = multispectral(id)
    print("image shape: ", m.shape)

    rgb = rgb_for_img(m)
    true_msk = true_mask_for_img(m, id)
    pred_msk = prediction_mask_for_img(m, model)

    for i in range(CLASSES):
        # create the plot
        plt.figure(figsize=(20,20))

        # plot the original image
        ax1 = plt.subplot(131)
        ax1.set_title('image ID: ' + id)
        ax1.imshow(stretch_n(rgb))

        # plot image of true mask
        ax2 = plt.subplot(132)
        ax2.set_title(CLASS_LIST[i] +" actual")
        ax2.imshow(true_msk[i], cmap=plt.get_cmap('gray'))

        # plot image of predicted mask
        ax3 = plt.subplot(133)
        ax3.set_title(CLASS_LIST[i] +" predicted")
        ax3.imshow(pred_msk[i], cmap=plt.get_cmap('gray'))

        # save the plot
        plt.savefig(inDir + '/kaggle/figures/detailed/' + id + '-' + CLASS_LIST[i])

def check_single(id):
    check_predict_simple(id)
    check_predict_detailed(id)    

def check_all():
    dataFrame = pd.read_csv(inDir + '/train_wkt_v4.csv')

    ids = sorted(dataFrame.ImageId.unique())
    print("image ids: ", ids)

    for id in ids:
        check_predict_simple(id)

if __name__ == "__main__":
    prepare_training_data()

    train_net()

    check_single('6120_2_2')

    # check_all()