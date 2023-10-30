import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
from shapely.wkt import loads as wkt_loads
import tifffile as tiff
import os
import random
from keras import backend as keras_backend
from sklearn.metrics import jaccard_score
from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D, Dropout, concatenate
from keras.optimizers import legacy
from keras.callbacks import ModelCheckpoint
import gc

# todo - rework code to use tensorflow format instead of theano
keras_backend.set_image_data_format('channels_first')

CLASSES = 10
CHANNELS = 8
INPUT_SIZE = 160
EPSILON = 1e-12

CLASS_LIST = [
    "Buildings", "Misc Manmade structures", "Road",
    "Track", "Trees", "Crops", "Waterway", "Standing water",
    "Vehicle Large", "Vehicle Small"
]
AVERAGE_CLASS_FREQUENCIES = [
    0.03,
    0.009,
    0.008,
    0.03,
    0.1,
    0.25,
    0.005,
    0.001,
    0.00003,
    0.00001
]

inDir = './'
os.makedirs(inDir + 'kaggle/data', exist_ok=True)
os.makedirs(inDir + 'kaggle/figures', exist_ok=True)
# os.makedirs(inDir + 'kaggle/msk', exist_ok=True)
# os.makedirs(inDir + 'kaggle/weights', exist_ok=True)
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
    intersection = keras_backend.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = keras_backend.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + EPSILON) / (sum_ - intersection + EPSILON)

    return keras_backend.mean(jac)

# neural op
def jaccard_coef_int(y_true, y_pred):
    y_pred_pos = keras_backend.round(keras_backend.clip(y_pred, 0, 1))
    intersection = keras_backend.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = keras_backend.sum(y_true + y_pred_pos, axis=[0, -1, -2])
    jac = (intersection + EPSILON) / (sum_ - intersection + EPSILON)

    return keras_backend.mean(jac)

# neural op
def stick_images_together():
    print("stick images together")
    s = 835

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
            for z in range(CLASSES):
                y[s * i:s * i + s, s * j:s * j + s, z] = generate_mask_for_image_and_class(
                    (img.shape[0], img.shape[1]), id, z + 1, gridSizes, dataFrame)[:s, :s]

    print("mask data mapped to range: ", np.amin(y), np.amax(y))

    np.save(inDir + '/kaggle/data/input_training_%d' % CLASSES, x)
    np.save(inDir + '/kaggle/data/output_training_%d' % CLASSES, y)

    gc.collect()

# data op
def get_patches(img, msk, amt=2000):
    x_max, y_max = img.shape[0] - INPUT_SIZE, img.shape[1] - INPUT_SIZE

    inputs, outputs, matched_classes = [], [], []
    # tr = [0.4, 0.1, 0.1, 0.15, 0.3, 0.95, 0.1, 0.05, 0.001, 0.005]

    for i in range(amt):
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
            if ratio >= AVERAGE_CLASS_FREQUENCIES[j]:
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

    inputs = 2 * np.transpose(inputs, (0, 3, 1, 2)) - 1
    outputs = np.transpose(outputs, (0, 3, 1, 2))
    # inputs, outputs = 2 * np.transpose(inputs, (0, 3, 1, 2)) - 1, np.transpose(outputs, (0, 3, 1, 2))
    print (inputs.shape, outputs.shape, np.amax(inputs), np.amin(inputs), np.amax(outputs), np.amin(outputs))

    gc.collect()

    return inputs, outputs

# neural op
def make_validation_set():
    print ("let's pick some samples for validation")
    img = np.load(inDir + '/kaggle/data/input_training_%d.npy' % CLASSES)
    mask = np.load(inDir + '/kaggle/data/output_training_%d.npy' % CLASSES)
    x, y = get_patches(img, mask, amt=2000)

    np.save(inDir + '/kaggle/data/input_validation_%d' % CLASSES, x)
    np.save(inDir + '/kaggle/data/output_validation_%d' % CLASSES, y)

    gc.collect()

# neural op
def jaccard_loss(y_true, y_pred):
    return -jaccard_coef(y_true, y_pred)

# neural op
def get_unet():
    inputs = Input((CHANNELS, INPUT_SIZE, INPUT_SIZE))
    conv1 = Convolution2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_first')(inputs)
    conv1 = Convolution2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_first')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2),data_format='channels_first')(conv1)
    
    conv2 = Convolution2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_first')(pool1)
    conv2 = Convolution2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_first')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2),data_format='channels_first')(conv2)
    
    conv3 = Convolution2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_first')(pool2)
    conv3 = Convolution2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_first')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2),data_format='channels_first')(conv3)
    
    conv4 = Convolution2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_first')(pool3)
    conv4 = Convolution2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_first')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2),data_format='channels_first')(drop4)

    conv5 = Convolution2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_first')(pool4)
    conv5 = Convolution2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_first')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Convolution2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_first')(UpSampling2D(size = (2,2),data_format='channels_first')(drop5))
    merge6 = concatenate([drop4,up6], axis = 1)
    conv6 = Convolution2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_first')(merge6)
    conv6 = Convolution2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_first')(conv6)

    up7 = Convolution2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_first')(UpSampling2D(size = (2,2),data_format='channels_first')(conv6))
    merge7 = concatenate([conv3,up7], axis = 1)
    conv7 = Convolution2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_first')(merge7)
    conv7 = Convolution2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_first')(conv7)

    up8 = Convolution2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_first')(UpSampling2D(size = (2,2),data_format='channels_first')(conv7))
    merge8 = concatenate([conv2,up8], axis = 1)
    conv8 = Convolution2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_first')(merge8)
    conv8 = Convolution2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_first')(conv8)

    up9 = Convolution2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_first')(UpSampling2D(size = (2,2),data_format='channels_first')(conv8))
    merge9 = concatenate([conv1,up9], axis = 1)
    conv9 = Convolution2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_first')(merge9)
    conv9 = Convolution2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_first')(conv9)
    conv9 = Convolution2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_first')(conv9)
    
    conv10 = Convolution2D(CLASSES, (1, 1),strides=1, activation = 'sigmoid',data_format='channels_first')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer=legacy.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=[jaccard_coef, jaccard_coef_int, 'accuracy'])

    return model

# neural op
def calc_jacc(model):
    print("calculate jaccard score")
    img = np.load(inDir + '/kaggle/data/input_validation_%d.npy' % CLASSES)
    msk = np.load(inDir + '/kaggle/data/output_validation_%d.npy' % CLASSES)

    prd = model.predict(img, batch_size=4)
    print ("prediction shape: ", prd.shape, " expected shape: ", msk.shape)
    avg, trs = [], []

    for i in range(CLASSES):
        t_msk = msk[:, i, :, :]
        t_prd = prd[:, i, :, :]
        t_msk = t_msk.reshape(msk.shape[0] * msk.shape[2], msk.shape[3])
        t_prd = t_prd.reshape(msk.shape[0] * msk.shape[2], msk.shape[3])

        m, b_tr = 0, 0
        for j in range(CLASSES):
            tr = j / CLASSES
            pred_binary_mask = t_prd > tr

            jk = jaccard_score(t_msk, pred_binary_mask, average='micro')

            if jk > m:
                m = jk
                b_tr = tr
        
        avg.append(m)
        trs.append(b_tr)

    score = sum(avg) / CLASSES

    gc.collect()

    return score, trs

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

# neural op
def train_net():
    print ("start train net")
    x_val, y_val = np.load(inDir + '/kaggle/data/input_validation_%d.npy' % CLASSES), np.load(inDir + '/kaggle/data/output_validation_%d.npy' % CLASSES)
    img = np.load(inDir + '/kaggle/data/input_training_%d.npy' % CLASSES)
    msk = np.load(inDir + '/kaggle/data/output_training_%d.npy' % CLASSES)

    model = get_unet()

    # for visualization
    model.save("keras_model.h5")

    # TODO: reenable in the future 
    #model.load_weights('../input/trained-weight/unet_10_jk0.7565')
    #model_checkpoint = ModelCheckpoint('unet_tmp.hdf5', monitor='loss', save_best_only=True)

    for i in range(1):
        x_trn, y_trn = get_patches(img, msk)
        model.fit(x_trn, y_trn, batch_size=64, epochs=10, verbose=1, shuffle=True,
                  callbacks=[], validation_data=(x_val, y_val))
        
        del(x_trn, y_trn)
        gc.collect()

        score, trs = calc_jacc(model)
        print('jacc '+ str(score))

        # TODO: reenable in the future 
        # model.save_weights(inDir +'/kaggle/working/unet_10_jk%.4f' % score)

    return model, score, trs

def predict_id(id, model, trs):
    img = multispectral(id)
    x = stretch_n(img)

    cnv = np.zeros((960, 960, CHANNELS)).astype(np.float32)
    prd = np.zeros((CLASSES, 960, 960)).astype(np.float32)
    cnv[:img.shape[0], :img.shape[1], :] = x

    for i in range(0, 6):
        line = []
        for j in range(0, 6):
            line.append(cnv[i * INPUT_SIZE:(i + 1) * INPUT_SIZE, j * INPUT_SIZE:(j + 1) * INPUT_SIZE])

        x = 2 * np.transpose(line, (0, 3, 1, 2)) - 1
        tmp = model.predict(x, batch_size=4)
        for j in range(tmp.shape[0]):
            prd[:, i * INPUT_SIZE:(i + 1) * INPUT_SIZE, j * INPUT_SIZE:(j + 1) * INPUT_SIZE] = tmp[j]

    trs = [0.4, 0.1, 0.4, 0.3, 0.3, 0.5, 0.3, 0.6, 0.1, 0.1]
    for i in range(CLASSES):
        prd[i] = prd[i] > trs[i]

    return prd[:, :img.shape[0], :img.shape[1]]

def check_predict(model, id='6120_2_3'):
    print("check_predict")

    msk = predict_id(id, model, [0.4, 0.1, 0.4, 0.3, 0.3, 0.5, 0.3, 0.6, 0.1, 0.1])
    m = multispectral(id)
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
        plt.savefig(inDir + '/kaggle/figures/' + CLASS_LIST[i])

if __name__ == "__main__":
    stick_images_together()

    make_validation_set()

    model, score, trs = train_net()

    check_predict(model, '6120_2_2')