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
from keras.optimizers import legacy
import gc

# these are the classes defined in the data file
ORIGINAL_CLASS_LIST = [
    "Buildings", "Misc Manmade structures", "Road",
    "Track", "Trees", "Crops", "Waterway", "Standing water",
    "Vehicle Large", "Vehicle Small"
]

# mapping from original class index to new index
CLASS_INDEX_MAP = {
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

# these are the classes we will actually use
CLASS_LIST = [
    "Buildings", "Misc Manmade structures", "Road", "Track",
    "Trees", "Crops", "Water"
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

ORIGINAL_CLASSES = len(ORIGINAL_CLASS_LIST)
CLASSES = len(CLASS_LIST)
CHANNELS = 8
TRAINING_CYCLES = 3
INPUT_SIZE = 160
BATCH_SIZE = 64
EPSILON = 1e-12

inDir = './'
os.makedirs(inDir + 'kaggle/data', exist_ok=True)
os.makedirs(inDir + 'kaggle/figures', exist_ok=True)
# os.makedirs(inDir + 'kaggle/msk', exist_ok=True)
os.makedirs(inDir + 'kaggle/weights', exist_ok=True)
# os.makedirs(inDir + 'kaggle/subm', exist_ok=True)

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

def stretch_n(bands, lower_percent=1, higher_percent=99):
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
def stick_images_together():
    print("stick images together")
    s = 835

    # if necessary files already exist, skip this step
    if os.path.isfile(inDir + '/kaggle/data/input_training_%d.npy' % CLASSES) and os.path.isfile(inDir + '/kaggle/data/output_training_%d.npy' % CLASSES):
        print("data already prepared, skipping")
        return


    x = np.zeros((5 * s, 5 * s, CHANNELS))
    y = np.zeros((5 * s, 5 * s, CLASSES))

    dataFrame = pd.read_csv(inDir + '/train_wkt_v4.csv')
    gridSizes = pd.read_csv(inDir + '/grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)

    ids = sorted(dataFrame.ImageId.unique())
    print("image ids: ", ids)

    for i in range(5):
        for j in range(5):
            id = ids[5 * i + j]

            img = multispectral(id)
            img = stretch_n(img)
            print (id, "input data shape: ", img.shape,  "data mapped to range: ", np.amin(img), np.amax(img))
            x[s * i:s * i + s, s * j:s * j + s, :] = img[:s, :s, :]
            
            for z_org in range(ORIGINAL_CLASSES):
                # remap original index
                z = CLASS_INDEX_MAP[z_org]
                if z == -1:
                    continue

                y[s * i:s * i + s, s * j:s * j + s, z] = generate_mask_for_image_and_class(
                    (img.shape[0], img.shape[1]), id, z + 1, gridSizes, dataFrame)[:s, :s]

    print("mask data mapped to range: ", np.amin(y), np.amax(y))

    np.save(inDir + '/kaggle/data/input_training_%d' % CLASSES, x)
    np.save(inDir + '/kaggle/data/output_training_%d' % CLASSES, y)

    gc.collect()

# data op
def get_patches(img, msk, amt):
    x_max, y_max = img.shape[0] - INPUT_SIZE, img.shape[1] - INPUT_SIZE

    inputs, outputs, matched_classes = [], [], []
    # tr = [0.4, 0.1, 0.1, 0.15, 0.3, 0.95, 0.1, 0.05, 0.001, 0.005]

    while len(inputs) < amt:
        # select a starting point
        x_start = random.randint(0, x_max)
        y_start = random.randint(0, y_max)

        # create patch of size (INPUT_SIZE x INPUT_SIZE)
        input_patch = img[x_start:x_start + INPUT_SIZE, y_start:y_start + INPUT_SIZE]
        output_patch = msk[x_start:x_start + INPUT_SIZE, y_start:y_start + INPUT_SIZE]

        for j in range(CLASSES):
            # calculate how many pixels with this class are in the patch
            class_pixels = np.sum(output_patch[:, :, j])
            # calculate ratio of pixels with this class in the patch
            ratio = class_pixels / (INPUT_SIZE ** 2)
            # if the ratio is good enough, use it for validation
            if ratio >= AVERAGE_CLASS_FREQUENCIES[CLASS_LIST[j]]:
                # perform a horizontal flip with probability 0.5
                if random.uniform(0, 1) > 0.5:
                    input_patch = input_patch[::-1]
                    output_patch = output_patch[::-1]
                # perform a vertical flip with probability 0.5
                if random.uniform(0, 1) > 0.5:
                    input_patch = input_patch[:, ::-1]
                    output_patch = output_patch[:, ::-1]

                inputs.append(input_patch)
                outputs.append(output_patch)
                matched_classes.append(CLASS_LIST[j])

                # prevent same patch getting triggered by different classes 
                break

    inputs = 2 * np.array(inputs) - 1
    outputs = np.array(outputs)
    print ('inputs', inputs.shape, np.amin(inputs), np.amax(inputs))
    print ('outputs', outputs.shape, np.amin(outputs), np.amax(outputs))

    gc.collect()

    return inputs, outputs

# neural op
def evaluate_jacc(model, img, msk):
    print("evaluate predictions by calculating jaccard score")
    # get some new patches to calculate the jaccard score on new data
    x_val, y_val = get_patches(img, msk, BATCH_SIZE * 30)
    y_pred = model.predict(x_val, batch_size=4)

    print ("prediction shape: ", y_pred.shape, " expected shape: ", y_val.shape)

    score = jaccard_coef_int(y_val, y_pred).numpy()

    del(x_val, y_val, y_pred)
    gc.collect()

    return score

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

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=legacy.Adam(learning_rate=0.001), loss=jaccard_loss, metrics=[jaccard_coef_int])

    return model

# neural op
def train_net():
    print ("train net")
    img = np.load(inDir + '/kaggle/data/input_training_%d.npy' % CLASSES)
    msk = np.load(inDir + '/kaggle/data/output_training_%d.npy' % CLASSES)

    model = get_unet()

    # for visualization
    model.save("keras_model.h5")

    # TODO: reenable in the future 
    #model.load_weights('../input/trained-weight/unet_10_jk0.7565')

    for i in range(1):
        x_trn, y_trn = get_patches(img, msk, BATCH_SIZE * 30)
        # TODO: split patches in 2 parts: i.e. 80% for training, 20% for validation and pass them
        model.fit(x_trn, y_trn, batch_size=BATCH_SIZE, epochs=10, verbose=1, shuffle=True)
        
        del(x_trn, y_trn)
        gc.collect()

        score = evaluate_jacc(model, img, msk)
        print('jacc '+ str(score))

    model.save_weights(inDir +'/kaggle/weights/unet_%d' % CLASSES)

def predict_on_img(img, model):
    print("making a prediction on given image")
    x = stretch_n(img)

    cnv = np.zeros((INPUT_SIZE * 6, INPUT_SIZE * 6, CHANNELS)).astype(np.float32)
    prd = np.zeros((INPUT_SIZE * 6, INPUT_SIZE * 6, CLASSES)).astype(np.float32)
    cnv[:img.shape[0], :img.shape[1], :] = x

    # this processes the input image in INPUT_SIZE*INPUT_SIZE patches
    # since the neural network can only ingest them this way
    for i in range(0, 6):
        line = []
        for j in range(0, 6):
            line.append(cnv[i * INPUT_SIZE:(i + 1) * INPUT_SIZE, j * INPUT_SIZE:(j + 1) * INPUT_SIZE])

        x = 2 * np.array(line) - 1
        tmp = model.predict(x, batch_size=4)

        for j in range(tmp.shape[0]):
            prd[i * INPUT_SIZE:(i + 1) * INPUT_SIZE, j * INPUT_SIZE:(j + 1) * INPUT_SIZE, :] = tmp[j]

    # we want class to be first dimension (makes drawing easier)
    pred_by_class = prd.transpose(2, 0, 1)

    # resize the predictions to match the image size
    return pred_by_class[:, :img.shape[0], :img.shape[1]]

def check_predict(id='6120_2_3'):
    model = get_unet()
    model.load_weights(inDir +'/kaggle/weights/unet_%d' % CLASSES)
    print("check_predict")

    m = multispectral(id)
    msk = predict_on_img(m, model)
    print(m.shape)

    # this creates RGB image from multispectral data
    img = np.zeros((m.shape[0],m.shape[1],3))
    img[:,:,0] = m[:,:,4] #red channel
    img[:,:,1] = m[:,:,2] #green channel
    img[:,:,2] = m[:,:,1] #blue channel

    for i in range(CLASSES):
        # plot the original image
        plt.figure(figsize=(20,20))
        ax1 = plt.subplot(131)
        ax1.set_title('image ID: ' + id)
        ax1.imshow(stretch_n(img))
        # plot image of predictions
        ax2 = plt.subplot(132)
        ax2.set_title("predict "+ CLASS_LIST[i] +" pixels")
        ax2.imshow(msk[i], cmap=plt.get_cmap('gray'))
        plt.savefig(inDir + '/kaggle/figures/' + id + '-' + CLASS_LIST[i])

if __name__ == "__main__":
    stick_images_together()

    train_net()

    check_predict('6120_2_2')