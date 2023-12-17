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
import time

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

# boundary at which we say a class "exists" based on ratio
CLASS_EXISTS_BOUNDARY = {
    "Buildings": 0.01,
    "Misc Manmade structures": 0.0001,
    "Road": 0.0001,
    "Track": 0.0001,
    "Trees": 0.1,
    "Crops": 0.1,
    "Water": 0.001,
}

# how many samples to take for single patch with this class
CLASS_SAMPLE_RATE = {
    "Buildings": 2,
    "Misc Manmade structures": 4,
    "Road": 2,
    "Track": 4,
    "Trees": 1,
    "Crops": 1,
    "Water": 4,
}

CLASS_COLORS = {
    "Buildings": [0xAA, 0xAA, 0xAA, 0],
    "Misc Manmade structures": [0x66, 0x66, 0x66, 0],
    "Road": [0xB3, 0x58, 0x06, 0],
    "Track":[0xDF, 0xC2, 0x7D, 0],
    "Trees": [0x1B, 0x78, 0x37, 0],
    "Crops":[0xA6, 0xDB, 0xA0, 0],
    "Water": [0x74, 0xAD, 0xD1, 0],
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
CHANNELS = 11
TRAINING_CYCLES = 3
EPOCHS = 10
INPUT_SIZE = 128
BATCH_SIZE = 64
INITIAL_LEARNING_RATE = 0.001
EPSILON = 1e-12

inDir = './'
logsDir = inDir + 'kaggle/logs/' + str(time.time())
os.makedirs(inDir + 'kaggle/data', exist_ok=True)
os.makedirs(inDir + 'kaggle/figures/simple', exist_ok=True)
os.makedirs(inDir + 'kaggle/figures/detailed', exist_ok=True)
os.makedirs(inDir + 'kaggle/weights', exist_ok=True)
os.makedirs(inDir + logsDir, exist_ok=True)

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

def CCCI_index(m):
    R = m[:,:,4]
    RE  = m[:,:,5] 
    NIR = m[:,:,7]
    # canopy chloropyll content index
    CCCI = (NIR-RE)/(NIR+RE)*(NIR-R)/(NIR+R)

    return CCCI

def EVI_index(m):
    R = m[:,:,4]
    B = m[:,:,1]
    NIR = m[:,:,7]
    L = 1
    C1 = 6
    C2 = 7.5
    G = 2.5
    EVI = G * (NIR - R) / ((NIR + C1) * (R - C2) * (B + L))

    return EVI

# data op
def multispectral(image_id):
    filename = os.path.join(inDir, 'sixteen_band', '{}_M.tif'.format(image_id))
    img = tiff.imread(filename)
    img = np.rollaxis(img, 0, 3)

    return img

def infrared(image_id):
    filename = os.path.join(inDir, 'sixteen_band', '{}_A.tif'.format(image_id))
    img = tiff.imread(filename)
    img = np.rollaxis(img, 0, 3)

    return img

def panchromatic(image_id):
    filename = os.path.join(inDir, 'sixteen_band', '{}_P.tif'.format(image_id))
    img = tiff.imread(filename)

    return img

def get_channels(image_id):
    # get multispectral channels
    m = multispectral(image_id)
    # get panchromatic channels
    p = panchromatic(image_id)
    # resize panchromatic channels to match multispectral
    p = cv2.resize(p, (m.shape[1], m.shape[0]))
    p = np.reshape(p, (p.shape[0], p.shape[1], 1))
    # get CCI channel
    ccci = CCCI_index(m)
    ccci = np.reshape(ccci, (ccci.shape[0], ccci.shape[1], 1))
    # get EVI channel
    evi = EVI_index(m)
    evi = np.reshape(evi, (evi.shape[0], evi.shape[1], 1))

    # merge all channels
    img = np.concatenate((m, p, ccci, evi), axis=2)

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

def jaccard_coef(y_true, y_pred):
    intersection = keras_backend.sum(y_true * y_pred, axis=[0,1,2])
    sum_ = keras_backend.sum(y_true + y_pred, axis=[0,1,2])
    jac = (intersection + EPSILON) / (sum_ - intersection + EPSILON)

    return keras_backend.mean(jac)

def jaccard_coef_int(y_true, y_pred):
    y_pred_pos = keras_backend.round(keras_backend.clip(y_pred, 0, 1))
    intersection = keras_backend.sum(y_true * y_pred_pos, axis=[0,1,2])
    sum_ = keras_backend.sum(y_true + y_pred_pos, axis=[0,1,2])
    jac = (intersection + EPSILON) / (sum_ - intersection + EPSILON)

    return keras_backend.mean(jac)

def jaccard_loss(y_true, y_pred):
    return -jaccard_coef(y_true, y_pred)

def dice_coef(y_true, y_pred, smooth=100):
    y_true_f = keras_backend.flatten(y_true)
    y_pred_f = keras_backend.flatten(y_pred)
    intersection = keras_backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (keras_backend.sum(y_true_f) + keras_backend.sum(y_pred_f) + smooth)

def dice_coef_multilabel(y_true, y_pred):
    total=0

    for index in range(CLASSES):
        total += dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])

    return total / CLASSES

def dice_loss(y_true, y_pred):
    return -dice_coef_multilabel(y_true, y_pred)

def combined_loss(y_true, y_pred):
    return 0.5 * dice_loss(y_true, y_pred) + 0.5 * jaccard_loss(y_true, y_pred)

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
        img = get_channels(id)
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

def get_patch_samples(input_patch, output_patch):
    sample_rate = 1
    total_pixels = INPUT_SIZE ** 2

    for i in range(CLASSES - 1):
        class_pixels = np.sum(output_patch[:, :, i])
        ratio = class_pixels / total_pixels
        class_name = CLASS_LIST[i]

        if ratio >= CLASS_EXISTS_BOUNDARY[class_name]:
            sample_rate = max(sample_rate, CLASS_SAMPLE_RATE[class_name])

    # normal, flip x, flip y, flip xy
    patch_variants = [
        (input_patch, output_patch),
        (input_patch[::-1], output_patch[::-1]),
        (input_patch[:, ::-1], output_patch[:, ::-1]),
        (input_patch[::-1, ::-1], output_patch[::-1, ::-1]),
    ]

    return random.sample(patch_variants, sample_rate)

# data op
def generate_patches(inputs, outputs, amount):
    input_patches, output_patches = [], []

    while len(input_patches) < amount:
        # rotate over images used for patch extraction
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

        samples = get_patch_samples(input_patch, output_patch)

        for (in_sample, out_sample) in samples:
            input_patches.append(in_sample)
            output_patches.append(out_sample)


    input_patches = 2 * np.array(input_patches[:amount]) - 1
    output_patches = np.array(output_patches[:amount])
    print ('input patches', input_patches.shape, np.amin(input_patches), np.amax(input_patches))
    print ('output patches', output_patches.shape, np.amin(output_patches), np.amax(output_patches))

    gc.collect()

    return input_patches, output_patches

# neural op
def evaluate_jacc(model):
    print("evaluate predictions by calculating jaccard score")
    # get some new patches to calculate the jaccard score on new data
    x_val, y_val = get_patches(BATCH_SIZE * 30)
    y_pred = model.predict(x_val, batch_size=4)

    print ("prediction shape: ", y_pred.shape, " expected shape: ", y_val.shape)

    scores = []
    weights = []
    total_count = x_val.shape[0] * x_val.shape[1] * x_val.shape[2]

    for i in range(CLASSES):
        class_true = y_val[:, :, :, i]
        class_pred = y_pred[:, :, :, i]
        class_score = jaccard_coef_int(class_true, class_pred).numpy()
        scores.append(class_score)
        print('class score for', CLASS_LIST[i], class_score)

        class_count = np.sum(class_true)
        weights.append(class_count / total_count)

    score = sum(scores) / CLASSES
    print('average score for all classes', score)

    weighted_score = 0
    for i in range(CLASSES):
        weighted_score += scores[i] * weights[i]
    
    print('weighted score for all classes', weighted_score)

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
    p = Dropout(0.15)(p)

    return f, p

def upsample_block(x, conv_features, n_filters):
    # upsample
    x = Convolution2DTranspose(n_filters, 3, 2, padding="same")(x)
    x = concatenate([x, conv_features])
    x = Dropout(0.15)(x)
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
        initial_learning_rate=INITIAL_LEARNING_RATE,
        decay_steps=150,
        decay_rate=0.96,
        staircase=False
    )

    # compile the model
    model.compile(optimizer=legacy.Adam(learning_rate=learning_rate_scheduler), loss=combined_loss, metrics=["accuracy", jaccard_coef_int, dice_coef_multilabel])

    return model

def get_patches(amount):
    inputs = np.load(inDir + '/kaggle/data/input_training_%d.npy' % CLASSES)
    outputs = np.load(inDir + '/kaggle/data/output_training_%d.npy' % CLASSES)
    x_trn, y_trn = generate_patches(inputs, outputs, amount)

    del(inputs, outputs)
    gc.collect()

    return (x_trn, y_trn)

# neural op
def train_net(reuse_prev=False):
    print ("train net")
    model = get_unet()

    # for visualization
    model.save("keras_model.h5")

    # for reusing previous weights
    if reuse_prev:
        model.load_weights(inDir +'/kaggle/weights_best/unet_%d' % CLASSES)

    for i in range(TRAINING_CYCLES):
        x_trn, y_trn = get_patches(BATCH_SIZE * 30)
        # create tensorboard for monitoring
        tensorboard = TensorBoard(log_dir=logsDir,update_freq='batch')
        # fit the model to the data
        model.fit(x_trn, y_trn, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, shuffle=True, callbacks=[tensorboard])
        
        del(x_trn, y_trn)
        gc.collect()

        evaluate_jacc(model)

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
            result[x][y][3] = 255 * distr[argmax] * 0.5
    
    return result

def check_predict_simple(id):
    print("check_predict (simple)")
    model = get_unet()
    model.load_weights(inDir +'/kaggle/weights/unet_%d' % CLASSES)

    img = get_channels(id)
    print("image shape: ", img.shape)

    rgb = rgb_for_img(img)
    true_msk = true_mask_for_img(img, id)
    pred_msk = prediction_mask_for_img(img, model)

    full_msk_true = full_mask(img, true_msk)
    full_msk_pred = full_mask(img, pred_msk)

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

    img = get_channels(id)
    print("image shape: ", img.shape)

    rgb = rgb_for_img(img)
    true_msk = true_mask_for_img(img, id)
    pred_msk = prediction_mask_for_img(img, model)

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
    # prepare_training_data()

    # train_net()

    check_single('6120_2_2')

    # check_all()