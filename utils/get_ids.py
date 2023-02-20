import numpy as np


def get_aligned_ids_nocs(masks, output_indices, class_ids, class_ids_predicted, scores, scores_predicted, depth):
  mask_out = []
  for p in range(masks.shape[-1]):
    mask = np.logical_and(masks[:, :, p], depth > 0)
    mask_out.append(mask)
  mask_out = np.array(mask_out)
  index_centers = []
  for m in range(mask_out.shape[0]):
    pos = np.where(mask_out[m,:,:])
    center_x = np.average(pos[0])
    center_y = np.average(pos[1])
    index_centers.append([center_x, center_y])
  new_masks = []
  new_ids = []
  new_scores = []
  index_centers = np.array(index_centers)
  if np.any(np.isnan(index_centers)):
    index_centers = index_centers[~np.any(np.isnan(index_centers), axis=1)]
  mask_out = np.array(mask_out)
  for l in range(len(output_indices)):
    point = output_indices[l]
    if len(output_indices) == 0:
      continue
    distances = np.linalg.norm(index_centers-point, axis=1)
    min_index = np.argmin(distances)
    if distances[min_index]<28:
      new_masks.append(mask_out[min_index, :,:])
      new_ids.append(class_ids[min_index])
      new_scores.append(scores[min_index])
    else: 
      new_masks.append(None)
      new_ids.append(class_ids_predicted[l])
      new_scores.append(scores_predicted[l])
  masks = np.array(new_masks)
  class_ids = np.array(new_ids)
  scores = np.array(new_scores)
  return masks, class_ids, scores
  
def get_ids_from_seg(seg_output, output_indices):
  category_seg_output = np.ascontiguousarray(seg_output.seg_pred.cpu().numpy())
  category_seg_output = np.argmax(category_seg_output[0], axis=0)
  class_ids_predicted = []
  for k in range(len(output_indices)):
    center = output_indices[k]
    class_ids_predicted.append(category_seg_output[center[0], center[1]])
  return class_ids_predicted