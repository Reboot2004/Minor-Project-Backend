import random
import numpy as np
from PIL import Image
import os
import tensorflow as tf
import cv2
from sklearn.mixture import GaussianMixture
import base64
import csv
from simple_lama_inpainting import SimpleLama
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import load_model
from utils import (
    select_sample_images,
    create_cell_descriptors_table,
    calculate_cell_descriptors,
    create_folders,
)

preprocessed_folder = 'uploads/'
segmentation_folder = 'segmentations/'
intermediate_folder = 'heatmaps/'
tables_folder = "tables/"
cell_descriptors_path = "cell_descriptors/cell_descriptors.csv"
saved_model_path = 'xception_model_81.h5'  # Replace with the path to your saved model

model = load_model(saved_model_path)

# add mapping from model output index to human-readable class
imgclasses = {0: "abnormal", 1: "normal"}


def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # Use Xception preprocessing to match the saved Xception model
    x = xception_preprocess(x)
    return x

def generate_grad_cam_plus_plus(img_path, model, last_conv_layer_name, classifier_layer_names):
    # image = edge_finding(img_path)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    img = tf.keras.applications.xception.preprocess_input(x)
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # print(grad_model)


    with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
        last_conv_output, preds = grad_model(img)
        # print(last_conv_output.shape)
        class_idx = np.argmax(preds[0])
        # max_value = tf.reduce_max(preds)

        # Reshape to get the desired shape (1,)
        # loss = tf.reshape(max_value, shape=(1,))
        loss = preds[:,0]
        # print('loss')
        # print(loss)
        grads = tape1.gradient(loss, last_conv_output)

        first_derivative = tf.exp(loss) * grads
        # print('grads')
        # print(first_derivative)

        second_derivative = tape2.gradient(grads, last_conv_output)
        second_derivative = tf.exp(loss) * second_derivative
        # print('grads2')
        # print(second_derivative)

    global_sum = tf.reduce_sum(first_derivative, axis=(0, 1, 2), keepdims=True)
    alpha_num = second_derivative
    alpha_denom = second_derivative * 2.0 + first_derivative * global_sum
    alphas = alpha_num / (alpha_denom + 1e-7)

    # print(alphas)


    weights = tf.maximum(0, global_sum)
    alpha_normalization_constant = tf.reduce_sum(alphas, axis=(0, 1), keepdims=True)
    alphas /= (alpha_normalization_constant + 1e-7)

    deep_linearization_weights = tf.reduce_sum(weights * alphas, axis=(0, 3))

    # Reshape the deep_linearization_weights to match the shape of last_conv_output
    deep_linearization_weights = tf.reshape(deep_linearization_weights, (1,7,7,-1))

    # print(deep_linearization_weights.shape)

    # Compute the CAM by taking a weighted sum of the convolutional layer output
    cam = tf.reduce_sum(deep_linearization_weights * last_conv_output, axis=3)



    # Normalize the CAM
    cam = tf.maximum(cam, 0)
    cam /= tf.reduce_max(cam)

    heatmap = tf.reduce_mean(cam, axis=0)  # Take mean along the channel axis
    # heatmap = tf.squeeze(cam)

    heatmap=heatmap.numpy()


    return heatmap


def GMM_abnormal_method(heatmap):
  heatmap = cv2.resize(heatmap, (224, 224))
  flat_heatmap = heatmap.flatten().reshape(-1, 1)

  # Define the number of clusters (segments)
  n_clusters = 4 # Adjust based on your requirements

  # Apply Gaussian Mixture Model clustering
  gmm = GaussianMixture(n_components=n_clusters, random_state=0)
  gmm.fit(flat_heatmap)
  labels = gmm.predict(flat_heatmap).reshape(heatmap.shape[:2])

  # Assign labels to the regions based on their intensity
  sorted_labels = np.argsort(gmm.means_.flatten())
  label_mapping = {sorted_labels[0]: 0, sorted_labels[1]: 1, sorted_labels[2]: 2,sorted_labels[3]: 3}
  labels_mapped = np.vectorize(label_mapping.get)(labels)

  colour_list=[[0,0,255],[128,0,0],[255,0,0],[255,0,0]]


  colors = np.array(colour_list)  # BGR format
  colored_labels = colors[labels_mapped]

  return labels_mapped,colored_labels

def GMM_normal_method(heatmap):
  heatmap = cv2.resize(heatmap, (224, 224))
  flat_heatmap = heatmap.flatten().reshape(-1, 1)

  # Define the number of clusters (segments)
  n_clusters = 4 # Adjust based on your requirements

  # Apply Gaussian Mixture Model clustering
  gmm = GaussianMixture(n_components=n_clusters, random_state=0)
  gmm.fit(flat_heatmap)
  labels = gmm.predict(flat_heatmap).reshape(heatmap.shape[:2])

  # Assign labels to the regions based on their intensity
  sorted_labels = np.argsort(gmm.means_.flatten())
  label_mapping = {sorted_labels[0]: 0, sorted_labels[1]: 1, sorted_labels[2]: 2,sorted_labels[3]: 3}
  labels_mapped = np.vectorize(label_mapping.get)(labels)

  colour_list=[[0,0,255],[128,0,0],[128,0,0],[255,0,0]]


  colors = np.array(colour_list)  # BGR format
  colored_labels = colors[labels_mapped]

  return labels_mapped,colored_labels

def create_nucelus(img,colored_segmentation_mask):
  mask=colored_segmentation_mask
  # Define the colors
  color_to_extract = [255, 0, 0]
  background_color = [0, 0, 255]

  # Create masks for the components and the background
  component_mask = np.all(mask == color_to_extract, axis=-1)
  background_mask = ~component_mask

  # Create an image with the extracted components in red and the background in blue
  result = np.zeros_like(mask)

  # cv2_imshow(result)
  result[component_mask] = color_to_extract
  result[background_mask] = background_color

  img= cv2.resize(img, (224,224))

  fgModel = np.zeros((1, 65), dtype="float")
  bgModel = np.zeros((1, 65), dtype="float")

  mask = np.zeros(result.shape[:2], np.uint8)
  mask[(result == [255, 0, 0]).all(axis=2)] = cv2.GC_PR_FGD  # Foreground
  mask[(result == [0, 0, 255]).all(axis=2)] = cv2.GC_PR_BGD  # Background

  # mask = np.mean(result, axis=2)
  # mask=mask.astype("uint8")

  rect = (0, 0, img.shape[1], img.shape[0])

  (mask, bgModel, fgModel) = cv2.grabCut(img, mask, rect, bgModel,
	fgModel, iterCount=10, mode=cv2.GC_INIT_WITH_MASK)

  output_image_1 = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

  # Replace black pixels with red and white pixels with blue
  output_image_1[mask == 2] = [0, 0, 255]  # Black to red
  output_image_1[mask == 3] = [255, 0, 0]  # White to blue

  return output_image_1

def create_colored_segmentation_mask(labels):
  colour_list=[0,0,0,0]

  # first_unique, second_unique, third_unique = find_unique_values(labels)
  colour_list[0]=[0,0,255]
  colour_list[1]=[255,0,0]
  colour_list[2]=[255,0,0]
  colour_list[3]=[255,0,0]
  # colour_list[4]=[255,0,0]

  colors = np.array(colour_list)  # BGR format
  colored_labels = colors[labels]

  return colored_labels

def create_background(img,heatmap,labels):
  colored_labels = create_colored_segmentation_mask(labels)
  mask=colored_labels
  # Define the colors
  color_to_extract = [255, 0, 0]
  background_color = [0, 0, 255]

  # Create masks for the components and the background
  component_mask = np.all(mask == color_to_extract, axis=-1)
  background_mask = ~component_mask

  # Create an image with the extracted components in red and the background in blue
  result = np.zeros_like(mask)
  result[component_mask] = color_to_extract
  result[background_mask] = background_color

  fgModel = np.zeros((1, 65), dtype="float")
  bgModel = np.zeros((1, 65), dtype="float")

  mask1 = np.zeros(result.shape[:2], np.uint8)
  mask1[(result == [255, 0, 0]).all(axis=2)] = cv2.GC_PR_FGD  # Foreground
  mask1[(result == [0, 0, 255]).all(axis=2)] = cv2.GC_PR_BGD  # Background

  # mask = np.mean(result, axis=2)
  # mask=mask.astype("uint8")

  rect = (1, 1, img.shape[1], img.shape[0])


  (mask1, bgModel, fgModel) = cv2.grabCut(img, mask1, rect, bgModel,
	fgModel, iterCount=10, mode=cv2.GC_INIT_WITH_MASK)


  output_image = np.zeros((mask1.shape[0], mask1.shape[1], 3), dtype=np.uint8)

  # Replace black pixels with red and white pixels with blue
  output_image[mask1 == 2] = [0, 0, 255]  # Black to red
  output_image[mask1 == 3] = [128, 0, 0]  # White to blue

  return output_image

def remove_nucleus(image, blue_mask):
  #expand the nucleus mask
  image1 = cv2.resize(image, (224,224))
  blue_mask1 = cv2.resize(blue_mask, (224,224))
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

def get_nucleus_mask(nucleus): #image_path, x, y
  # nucleus = cv2.imread(nucleus)
  # Convert image to HSV color space
  hsv_image = cv2.cvtColor(nucleus, cv2.COLOR_BGR2HSV)
  # Define lower and upper bounds for blue color in HSV
  lower_blue = np.array([100, 50, 50])
  upper_blue = np.array([130, 255, 255])
  # Create a mask for blue color
  blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
  return blue_mask #, image

def save_heatmap(heatmap,img_path,heatmap_path):
  img = cv2.imread(img_path)
  heatmap_1 = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

  heatmap_1 = np.uint8(255 * heatmap_1)

  heatmap_1 = cv2.applyColorMap(heatmap_1, cv2.COLORMAP_JET)

  superimposed_img = cv2.addWeighted(heatmap_1, 0.4,img, 0.6,  0)
  superimposed_img = np.uint8(superimposed_img)
  
  superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

  cv2.imwrite(heatmap_path,superimposed_img)


def cam_main(pixel_conversion):
  count=0

  return_dict_count = 1
  return_dict = {}
  selected_indices = select_sample_images()
  print('selected_indices')
  print(selected_indices)
  resized_shape = (224,224)
  cell_descriptors = [
      ["Image Name", "Nucleus Area", "Cytoplasm Area", "Nucleus to Cytoplasm Ratio"]
  ]

  image_files = [f for f in os.listdir(preprocessed_folder) if not f.startswith('.DS_Store')]

  for imagefile in image_files:
    if (
        "MACOSX".lower() in imagefile.lower()
        or "." == imagefile[0]
        or "_" == imagefile[0]
    ):
        print(imagefile)
        continue
    image_path = (
        preprocessed_folder + imagefile
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
    # img_path=input_folder+'/'+a

    # print(a)

    # count+=1

    # input_image = preprocess_image(img_path)

    heatmap = generate_grad_cam_plus_plus(image_path, model, 'block14_sepconv2_act', ['dense_1'])

    save_heatmap(heatmap,image_path,intermediate_path)
    

    pred_class = model.predict(preprocess_image(image_path))
    pred_class = pred_class.argmax(axis=1)[0]
    class_name = imgclasses.get(pred_class, str(pred_class))

    # print(pred_class)

    if pred_class == 0:
      labels,colored_segmentation_mask = GMM_abnormal_method(heatmap)
    else:
      labels,colored_segmentation_mask = GMM_normal_method(heatmap)

    image=cv2.imread(image_path)
    original_shape = image.shape
    image= cv2.resize(image, (224,224))

    nucleus= create_nucelus(image,colored_segmentation_mask)

    blue_mask = get_nucleus_mask(nucleus)

    expandedmask, result_cv2 = remove_nucleus(image, blue_mask)

    background=create_background(image,heatmap,labels)

    combined_mask = background & nucleus
  

    for i in range(combined_mask.shape[0]):
      for j in range(combined_mask.shape[1]):
          original_color = tuple(combined_mask[i, j])
          if original_color == (128,0,0):
              combined_mask[i, j] = np.array((255,0,0))
          elif original_color == (0,0,0):
              combined_mask[i, j] = np.array((128,0,0))

    combined_mask = cv2.resize(combined_mask, (224,224))

    cv2.imwrite(save_path,combined_mask)


    nucleus_area, cytoplasm_area, ratio = calculate_cell_descriptors(
            original_shape, resized_shape, pixel_conversion, combined_mask
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

    if count in selected_indices:
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
        # add predicted class so frontend can show it
        return_dict[f"class{return_dict_count}"] = class_name

        return_dict_count += 1

    count+=1

    print(count)

  with open(cell_descriptors_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(cell_descriptors)

  print(list(return_dict.keys()))

  return return_dict

    

# cam_main(0.2)


# ================= Single Image Entry Point =================
def cam_process_single_image(image_path: str, pixel_conversion: float):
  """
  Run the CAM pipeline on a single image file path.

  Inputs:
    - image_path: str, path to an image file (jpg/png/bmp...)
    - pixel_conversion: float, microns-per-pixel (or similar) conversion used in area calculation

  Returns:
    - return_dict: dict with base64-encoded 'image1', 'inter1', 'mask1', 'table1'
    - output_paths: dict with file paths for 'heatmap', 'mask', 'table'
  """
  # Ensure output folders exist
  folder_names = [
      "uploads",
      "heatmaps",
      "segmentations",
      "tables",
      "cell_descriptors",
  ]
  create_folders(folder_names)

  # Derive output file paths
  base_name = os.path.splitext(os.path.basename(image_path))[0].lower()
  intermediate_path = os.path.join(intermediate_folder, f"{base_name}_heatmap.png")
  save_path = os.path.join(segmentation_folder, f"{base_name}_mask.png")
  table_path = os.path.join(tables_folder, f"{base_name}_table.png")

  # Generate heatmap
  heatmap = generate_grad_cam_plus_plus(image_path, model, 'block14_sepconv2_act', ['dense_1'])
  save_heatmap(heatmap, image_path, intermediate_path)

  # Predict class to choose segmentation strategy
  pred_class = model.predict(preprocess_image(image_path)).argmax(axis=1)[0]
  class_name = imgclasses.get(pred_class, str(pred_class))
  if pred_class == 0:
    labels, colored_segmentation_mask = GMM_abnormal_method(heatmap)
  else:
    labels, colored_segmentation_mask = GMM_normal_method(heatmap)

  # Build combined mask
  image_cv = cv2.imread(image_path)
  if image_cv is None:
    raise ValueError(f"Can't read image: {image_path}")
  original_shape = image_cv.shape
  image_resized = cv2.resize(image_cv, (224, 224))

  nucleus = create_nucelus(image_resized, colored_segmentation_mask)
  background = create_background(image_resized, heatmap, labels)
  combined_mask = background & nucleus

  # Normalize colors to expected values
  for i in range(combined_mask.shape[0]):
    for j in range(combined_mask.shape[1]):
      original_color = tuple(combined_mask[i, j])
      if original_color == (128, 0, 0):
        combined_mask[i, j] = np.array((255, 0, 0))
      elif original_color == (0, 0, 0):
        combined_mask[i, j] = np.array((128, 0, 0))

  cv2.imwrite(save_path, combined_mask)

  # Compute descriptors and save table and CSV
  resized_shape = (224, 224)
  nucleus_area, cytoplasm_area, ratio = calculate_cell_descriptors(
      original_shape, resized_shape, pixel_conversion, combined_mask
  )

  # Save table image
  create_cell_descriptors_table(table_path, nucleus_area, cytoplasm_area, ratio)

  # Save CSV (header + single row)
  cell_descriptors = [
      ["Image Name", "Nucleus Area", "Cytoplasm Area", "Nucleus to Cytoplasm Ratio"],
      [base_name, nucleus_area, cytoplasm_area, ratio],
  ]
  with open(cell_descriptors_path, "w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerows(cell_descriptors)

  # Build return dict with base64-encoded artifacts and class label
  return_dict = {
      "image1": str(base64.b64encode(open(image_path, "rb").read()).decode("utf-8")),
      "inter1": str(base64.b64encode(open(intermediate_path, "rb").read()).decode("utf-8")),
      "mask1": str(base64.b64encode(open(save_path, "rb").read()).decode("utf-8")),
      "table1": str(base64.b64encode(open(table_path, "rb").read()).decode("utf-8")),
      "class1": class_name
  }

  output_paths = {"heatmap": intermediate_path, "mask": save_path, "table": table_path}
  return return_dict, output_paths


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(description="Run Grad-CAM++ pipeline on a single image.")
  parser.add_argument("--image", "-i", required=True, help="Path to the input image file")
  parser.add_argument("--magval", "-m", required=True, type=float, help="Pixel conversion value (e.g., 0.2)")
  args = parser.parse_args()

  result, paths = cam_process_single_image(args.image, args.magval)
  print("Processing complete. Outputs:")
  print(f"  Heatmap:      {paths['heatmap']}")
  print(f"  Segmentation: {paths['mask']}")
  print(f"  Table:        {paths['table']}")
