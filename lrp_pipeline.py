import cv2
import torch
import torch.nn as nn
import numpy as np
import torchvision
import os
import copy
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
from simple_lama_inpainting import SimpleLama
from PIL import Image
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib
import csv

matplotlib.use("Agg")

import base64

from utils import (
    select_sample_images,
    create_cell_descriptors_table,
    calculate_cell_descriptors,
)

preprocessed_folder = "uploads/"
intermediate_folder = "heatmaps/"
segmentation_folder = "segmentations/"
tables_folder = "tables/"
cell_descriptors_path = "cell_descriptors/cell_descriptors.csv"
imgclasses = {0: "abnormal", 1: "normal"}


def toconv(layers):
    newlayers = []
    for i, layer in enumerate(layers):
        if isinstance(layer, nn.Linear):
            newlayer = None
            if i == 0:
                m, n = 512, layer.weight.shape[0]
                newlayer = nn.Conv2d(m, n, 4)
                newlayer.weight = nn.Parameter(layer.weight.reshape(n, m, 4, 4))
            else:
                m, n = layer.weight.shape[1], layer.weight.shape[0]
                newlayer = nn.Conv2d(m, n, 1)
                newlayer.weight = nn.Parameter(layer.weight.reshape(n, m, 1, 1))
            newlayer.bias = nn.Parameter(layer.bias)
            newlayers += [newlayer]
        else:
            newlayers += [layer]
    return newlayers


def newlayer(layer, g):
    layer = copy.deepcopy(layer)
    try:
        layer.weight = nn.Parameter(g(layer.weight))
    except AttributeError:
        pass
    try:
        layer.bias = nn.Parameter(g(layer.bias))
    except AttributeError:
        pass
    return layer


def heatmap(R, sx, sy, intermediate_path):
    b = 10 * ((np.abs(R) ** 3.0).mean() ** (1.0 / 3))
    my_cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
    my_cmap[:, 0:3] *= 0.85
    my_cmap = ListedColormap(my_cmap)
    plt.figure(figsize=(sx, sy))
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.axis("off")
    plt.imshow(R, cmap=my_cmap, vmin=-b, vmax=b, interpolation="nearest")
    # plt.show()
    plt.savefig(intermediate_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def get_LRP_heatmap(image, L, layers, imgclasses, intermediate_path):
    img = np.array(image)[..., ::-1] / 255.0
    mean = torch.FloatTensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)  # torch.cuda
    std = torch.FloatTensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)  # torch.cuda
    X = (torch.FloatTensor(img[np.newaxis].transpose([0, 3, 1, 2]) * 1) - mean) / std

    A = [X] + [None] * L
    for l in range(L):
        A[l + 1] = layers[l].forward(A[l])

    scores = np.array(A[-1].cpu().data.view(-1))
    ind = np.argsort(-scores)
    for i in ind[:2]:
        print("%20s (%3d): %6.3f" % (imgclasses[i], i, scores[i]))

    T = torch.FloatTensor(
        (1.0 * (np.arange(2) == ind[0]).reshape([1, 2, 1, 1]))
    )  # SET FOR THE HIGHEST SCORE CLASS
    R = [None] * L + [(A[-1] * T).data]
    for l in range(1, L)[::-1]:
        A[l] = (A[l].data).requires_grad_(True)
        if isinstance(layers[l], torch.nn.MaxPool2d):
            layers[l] = torch.nn.AvgPool2d(2)
        if isinstance(layers[l], torch.nn.Conv2d) or isinstance(
            layers[l], torch.nn.AvgPool2d
        ):
            rho = lambda p: p + 0.25 * p.clamp(min=0)
            incr = lambda z: z + 1e-9  # USE ONLY THE GAMMA RULE FOR ALL LAYERS

            z = incr(newlayer(layers[l], rho).forward(A[l]))  # step 1
            # adding epsilon
            epsilon = 1e-9
            z_nonzero = torch.where(z == 0, torch.tensor(epsilon, device=z.device), z)
            s = (R[l + 1] / z_nonzero).data
            # s = (R[l+1]/z).data                                    # step 2
            (z * s).sum().backward()
            c = A[l].grad  # step 3
            R[l] = (A[l] * c).data  # step 4
        else:
            R[l] = R[l + 1]

    A[0] = (A[0].data).requires_grad_(True)
    lb = (A[0].data * 0 + (0 - mean) / std).requires_grad_(True)
    hb = (A[0].data * 0 + (1 - mean) / std).requires_grad_(True)

    z = layers[0].forward(A[0]) + 1e-9  # step 1 (a)
    z -= newlayer(layers[0], lambda p: p.clamp(min=0)).forward(lb)  # step 1 (b)
    z -= newlayer(layers[0], lambda p: p.clamp(max=0)).forward(hb)  # step 1 (c)

    # adding epsilon
    epsilon = 1e-9
    z_nonzero = torch.where(z == 0, torch.tensor(epsilon, device=z.device), z)
    s = (R[1] / z_nonzero).data  # step 2

    (z * s).sum().backward()
    c, cp, cm = A[0].grad, lb.grad, hb.grad  # step 3
    R[0] = (A[0] * c + lb * cp + hb * cm).data  # step 4
    heatmap(
        np.array(R[0][0].cpu()).sum(axis=0), 2, 2, intermediate_path
    )  # HEATMAPPING TO SEE LRP MAPS WITH NEW RULE
    return R[0][0].cpu()


def get_nucleus_mask_for_graphcut(R):
    res = np.array(R).sum(axis=0)
    # Reshape the data to a 1D array
    data_1d = res.flatten().reshape(-1, 1)
    n_clusters = 2
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    # kmeans.fit(data_1d)
    kmeans.fit(data_1d)
    # Step 4: Assign data points to clusters
    cluster_assignments = kmeans.labels_
    # Step 5: Reshape cluster assignments into a 2D binary matrix
    binary_matrix = cluster_assignments.reshape(128, 128)
    # Now, binary_matrix contains 0s and 1s, separating the data into two classes using K-Means clustering
    rel_grouping = np.zeros((128, 128, 3), dtype=np.uint8)
    rel_grouping[binary_matrix == 1] = [255, 0, 0]  # Main object (Blue)
    rel_grouping[binary_matrix == 2] = [128, 0, 0]  # Second label (Dark Blue)
    rel_grouping[binary_matrix == 0] = [0, 0, 255]  # Background (Red)
    return rel_grouping


def segment_nucleus(image, rel_grouping):  # clustered = rel_grouping

    # GET THE BOUNDING BOX FROM CLUSTERED
    blue_pixels = np.sum(np.all(rel_grouping == [255, 0, 0], axis=-1))
    red_pixels = np.sum(np.all(rel_grouping == [0, 0, 255], axis=-1))
    if red_pixels > blue_pixels:
        color = np.array([255, 0, 0])
    else:
        color = np.array([0, 0, 255])
    mask = cv2.inRange(rel_grouping, color, color)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_areas = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        contour_areas.append(cv2.contourArea(contour))
    contour_areas.sort()
    contour_areas = np.array(contour_areas)
    quartile_50 = np.percentile(contour_areas, 50)
    selected_contours = [
        contour for contour in contours if cv2.contourArea(contour) >= quartile_50
    ]
    x, y, w, h = cv2.boundingRect(np.concatenate(selected_contours))

    # APPLY GRABCUT
    fgModel = np.zeros((1, 65), dtype="float")
    bgModel = np.zeros((1, 65), dtype="float")
    mask = np.zeros(image.shape[:2], np.uint8)
    rect = (x, y, x + w, y + h)

    # IF BOUNDING BOX IS THE WHOLE IMAGE, THEN BOUNDING BOX METHOD WONT'T WORK -> SO USE INIT WITH MASK METHOD ITSELF
    if (x, y, x + w, y + h) == (0, 0, 128, 128):

        if (
            red_pixels > blue_pixels
        ):  # red is the dominant color and thus the background
            mask[(rel_grouping == [255, 0, 0]).all(axis=2)] = (
                cv2.GC_PR_FGD
            )  # Probable Foreground
            mask[(rel_grouping == [0, 0, 255]).all(axis=2)] = (
                cv2.GC_PR_BGD
            )  # Probable Background
        else:  # blue is the dominant color and thus the background
            mask[(rel_grouping == [0, 0, 255]).all(axis=2)] = (
                cv2.GC_PR_FGD
            )  # Probable Foreground
            mask[(rel_grouping == [255, 0, 0]).all(axis=2)] = (
                cv2.GC_PR_BGD
            )  # Probable Background

        (mask, bgModel, fgModel) = cv2.grabCut(
            image,
            mask,
            rect,
            bgModel,
            fgModel,
            iterCount=10,
            mode=cv2.GC_INIT_WITH_MASK,
        )

    # ELSE PASS THE BOUNDING BOX FOR GRABCUT
    else:
        (mask, bgModel, fgModel) = cv2.grabCut(
            image,
            mask,
            rect,
            bgModel,
            fgModel,
            iterCount=10,
            mode=cv2.GC_INIT_WITH_RECT,
        )

    # FORM THE COLORED SEGMENTATION MASK
    clean_binary_mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0
    ).astype("uint8")
    nucleus_segment = np.zeros((128, 128, 3), dtype=np.uint8)
    nucleus_segment[clean_binary_mask == 1] = [255, 0, 0]  # Main object (Blue)
    nucleus_segment[clean_binary_mask == 0] = [0, 0, 255]  # Background (Red)
    return nucleus_segment, clean_binary_mask


def remove_nucleus(image1, blue_mask1):  # image, blue_mask, x, y
    # expand the nucleus mask
    # image1 = cv2.resize(image, (128,128))
    # blue_mask1 = cv2.resize(blue_mask, (128,128))
    kernel = np.ones((5, 5), np.uint8)  # Adjust the kernel size as needed
    expandedmask = cv2.dilate(blue_mask1, kernel, iterations=1)
    simple_lama = SimpleLama()
    image_pil = Image.fromarray(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    mask_pil = Image.fromarray(expandedmask)
    result = simple_lama(image_pil, mask_pil)
    result_cv2 = np.array(result)
    result_cv2 = cv2.cvtColor(result_cv2, cv2.COLOR_RGB2BGR)
    # result_cv2 = cv2.resize(result_cv2, (x,y))
    return expandedmask, result_cv2


def get_final_mask(nucleus_removed_img, blue_mask, expanded_mask):
    # apply graphcut - init with rectangle (not mask approximation mask)
    fgModel = np.zeros((1, 65), dtype="float")
    bgModel = np.zeros((1, 65), dtype="float")

    rect = (1, 1, nucleus_removed_img.shape[1], nucleus_removed_img.shape[0])

    (mask, bgModel, fgModel) = cv2.grabCut(
        nucleus_removed_img,
        expanded_mask,
        rect,
        bgModel,
        fgModel,
        iterCount=20,
        mode=cv2.GC_INIT_WITH_RECT,
    )

    clean_binary_mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0
    ).astype("uint8")
    colored_segmentation_mask = np.zeros((128, 128, 3), dtype=np.uint8)
    colored_segmentation_mask[clean_binary_mask == 1] = [
        128,
        0,
        0,
    ]  # Main object (Blue)
    colored_segmentation_mask[clean_binary_mask == 0] = [0, 0, 255]  # Background (Red)
    colored_segmentation_mask[blue_mask > 0] = [255, 0, 0]
    return colored_segmentation_mask


def lrp_main(pixel_conversion):
    i = 0
    return_dict_count = 1
    return_dict = {}
    selected_indices = select_sample_images()
    resized_shape = (128, 128)
    cell_descriptors = [
        ["Image Name", "Nucleus Area", "Cytoplasm Area", "Nucleus to Cytoplasm Ratio"]
    ]

    for imagefile in os.listdir(preprocessed_folder):
        if (
            "MACOSX".lower() in imagefile.lower()
            or "." == imagefile[0]
            or "_" == imagefile[0]
        ):
            print(imagefile)
            continue
        image_path = (
            preprocessed_folder + os.path.splitext(imagefile)[0].lower() + ".jpg"
        )
        intermediate_path = (
            intermediate_folder
            + os.path.splitext(imagefile)[0].lower()
            + "_heatmap.png"
        )
        save_path = (
            segmentation_folder + os.path.splitext(imagefile)[0].lower() + "_mask.png"
        )
        table_path = (
            tables_folder + os.path.splitext(imagefile)[0].lower() + "_table.png"
        )

        # print(i, imagefile)
        image = cv2.imread(image_path)
        original_shape = image.shape

        image = cv2.resize(image, (128, 128))

        # MODEL SECTION STARTS FOR NEW MODEL
        vgg16 = torchvision.models.vgg16(pretrained=True)
        new_avgpool = nn.AdaptiveAvgPool2d(output_size=(4, 4))
        vgg16.avgpool = new_avgpool
        classifier_list = [
            nn.Linear(8192, vgg16.classifier[0].out_features)
        ]  # vgg16.classifier[0].out_features = 4096
        classifier_list += list(vgg16.classifier.children())[
            1:-1
        ]  # Remove the first and last layers
        classifier_list += [
            nn.Linear(vgg16.classifier[6].in_features, 2)
        ]  # vgg16.classifier[6].in_features = 4096
        vgg16.classifier = nn.Sequential(
            *classifier_list
        )  # Replace the model classifier

        PATH = "herlev_best_adam_vgg16_modified12_final.pth"
        checkpoint = torch.load(PATH, map_location=torch.device("cpu"))
        vgg16.load_state_dict(checkpoint)
        # vgg16.to(torch.device('cuda'))
        vgg16.eval()

        layers = list(vgg16._modules["features"]) + toconv(
            list(vgg16._modules["classifier"])
        )
        L = len(layers)
        # MODEL SECTION ENDS

        R = get_LRP_heatmap(image, L, layers, imgclasses, intermediate_path)

        rel_grouping = get_nucleus_mask_for_graphcut(R)

        nucleus_segment, clean_binary_mask = segment_nucleus(image, rel_grouping)

        expanded_mask, nucleus_removed_image = remove_nucleus(image, clean_binary_mask)

        colored_segmentation_mask = get_final_mask(
            nucleus_removed_image, clean_binary_mask, expanded_mask
        )

        cv2.imwrite(save_path, colored_segmentation_mask)

        nucleus_area, cytoplasm_area, ratio = calculate_cell_descriptors(
            original_shape, resized_shape, pixel_conversion, colored_segmentation_mask
        )
        cell_descriptors.append(
            [
                os.path.splitext(imagefile)[0].lower(),
                nucleus_area,
                cytoplasm_area,
                ratio,
            ]
        )

        create_cell_descriptors_table(table_path, nucleus_area, cytoplasm_area, ratio)

        if i in selected_indices:
            return_dict[f"image{return_dict_count}"] = str(
                base64.b64encode(open(image_path, "rb").read()).decode("utf-8")
            )
            return_dict[f"inter{return_dict_count}"] = str(
                base64.b64encode(open(intermediate_path, "rb").read()).decode("utf-8")
            )
            return_dict[f"mask{return_dict_count}"] = str(
                base64.b64encode(open(save_path, "rb").read()).decode("utf-8")
            )
            return_dict[f"table{return_dict_count}"] = str(
                base64.b64encode(open(table_path, "rb").read()).decode("utf-8")
            )
            return_dict_count += 1

        i += 1

        # Visualization
        # for im in [image, gt2, rel_grouping, nucleus_segment, clean_binary_mask*255, nucleus_removed_image, colored_segmentation_mask]:
        #   cv2_imshow(im)

    # write cell_descriptors list to csv file
    with open(cell_descriptors_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(cell_descriptors)

    return return_dict
