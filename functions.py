import matplotlib.patches as patches
import pandas as pd
from pycocotools.coco import COCO
from PIL import Image
import math


def load_coco_image(coco, coco_images_dir, image_id):
    return Image.open(coco_images_dir + '/' + coco.loadImgs(image_id)[0]['file_name'])


def get_xywh_bounding_boxes_and_labels_from_yolo_prediction(prediction, as_dataframe=False):
    xyxy = prediction.pandas().xyxy[0]
    xywh = prediction.pandas().xywh[0]
    labels = xyxy['name']
    bounding_boxes = pd.DataFrame({
        'xmin': xyxy['xmin'],
        'ymin': xyxy['ymin'],
        'width': xywh['width'],
        'height': xywh['height']
    })

    return (bounding_boxes, labels) if as_dataframe else (bounding_boxes.values.tolist(), labels.to_list())


def get_xywh_bounding_boxes_and_labels_from_coco(coco, image_id, as_dataframe=False):
    annotation_ids = coco.getAnnIds(imgIds=image_id)
    annotations = coco.loadAnns(annotation_ids)
    bounding_boxes = []
    labels = []
    for annotation in annotations:
        labels.append(coco.loadCats(annotation['category_id'])[0]['name'])
        bounding_boxes.append(annotation['bbox'])

    if as_dataframe:
        data = {
            'xmin': [i[0] for i in bounding_boxes],
            'xmax': [i[1] for i in bounding_boxes],
            'width': [i[2] for i in bounding_boxes],
            'height': [i[3] for i in bounding_boxes],
        }
        return pd.DataFrame(data), labels
    else:
        return bounding_boxes, labels


def draw_xywh_bounding_boxes(ax, bounding_boxes, labels, linewidth=2, edgecolor='r', facecolor='none'):
    for i in range(len(bounding_boxes)):
        box = bounding_boxes[i]
        label = labels[i]
        rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=linewidth, edgecolor=edgecolor, facecolor=facecolor)
        ax.add_patch(rect)
        ax.text(box[0], box[1] - 10, label, color=edgecolor)


def group_bounding_boxes_by_label(bounding_boxes, labels):
    bounding_boxes_by_label = {}
    for i in range(len(bounding_boxes)):
        if labels[i] not in bounding_boxes_by_label:
            bounding_boxes_by_label[labels[i]] = []
        bounding_boxes_by_label[labels[i]].append(bounding_boxes[i])

    return bounding_boxes_by_label


def bounding_boxes_iou(bounding_box_a, bounding_box_b):
    x_overlap = max(0, bounding_box_a[0] + bounding_box_a[2] - bounding_box_b[0], bounding_box_b[0] + bounding_box_b[2] - bounding_box_a[0])
    y_overlap = max(0, bounding_box_a[1] + bounding_box_a[3] - bounding_box_b[1], bounding_box_b[1] + bounding_box_b[3] - bounding_box_a[1])
    
    intersection = x_overlap * y_overlap
    union = bounding_box_a[2] * bounding_box_a[3] + bounding_box_b[2] * bounding_box_b[3] - intersection

    return intersection / union if union > 0 else 0


def distane_between_bounding_box_centres(bounding_box_a, bounding_box_b):
    deltaX = bounding_box_a[0] + bounding_box_a[2] / 2 - bounding_box_b[0] - bounding_box_b[2] / 2
    deltaY = bounding_box_a[1] + bounding_box_a[3] / 2 - bounding_box_b[1] - bounding_box_b[3] / 2
    return math.sqrt(deltaX ** 2 + deltaY ** 2)


def get_results_by_label(bounding_boxes_by_label = None, predicted_bounding_boxes_by_label = None, bounding_boxes = None, labels = None, predicted_bounding_boxes = None, predicted_labels = None):
    if bounding_boxes_by_label is None and (bounding_boxes is None or labels is None):
        return None
    
    if predicted_bounding_boxes_by_label is None and (predicted_bounding_boxes is None or predicted_labels is None):
        return None
    
    if bounding_boxes_by_label is None:
        bounding_boxes_by_label = group_bounding_boxes_by_label(bounding_boxes, labels)

    if predicted_bounding_boxes_by_label is None:
        predicted_bounding_boxes_by_label = group_bounding_boxes_by_label(predicted_bounding_boxes, predicted_labels)


    results_by_label = {}
    for label in set(bounding_boxes_by_label.keys()).union(set(predicted_bounding_boxes_by_label.keys())):
        results_by_label[label] = {
            'true positives': [],   # IoU > 0 for closest (closest center-to-center) ground truth bounding box
            'false positives': [],  # IoU == 0 for closest (closest center-to-center) ground truth bound box
            'false negatives': []   # Ground truth bound box has IoU = 0 for all predicted bounding boxes
        }

    for label in predicted_bounding_boxes_by_label:
        for predicted_bounding_box in predicted_bounding_boxes_by_label[label]:
            if label not in bounding_boxes_by_label:
                results_by_label[label]['false positives'].append(predicted_bounding_box)
                continue

            closest_bounding_box = None
            distance_to_closest_bounding_box = math.inf
            for bounding_box in bounding_boxes_by_label[label]:
                distance = distane_between_bounding_box_centres(bounding_box, predicted_bounding_box)
                if distance < distance_to_closest_bounding_box:
                    closest_bounding_box = bounding_box
                    distance_to_closest_bounding_box = distance

            iou = bounding_boxes_iou(predicted_bounding_box, closest_bounding_box)
            if iou == 0:
                results_by_label[label]['false positives'].append(predicted_bounding_box)
            else:
                results_by_label[label]['true positives'].append({'bounding_box': predicted_bounding_box, 'iou': iou})

    for label in bounding_boxes_by_label:
        for bounding_box in bounding_boxes_by_label[label]:
            if label not in predicted_bounding_boxes_by_label:
                results_by_label[label]['false negatives'].append(bounding_box)
                continue

            for predicted_bounding_box in predicted_bounding_boxes_by_label[label]:
                if bounding_boxes_iou(bounding_box, predicted_bounding_box) > 0:
                    break
            else:
                results_by_label[label]['false negatives'].append(bounding_box)

    return results_by_label

            

            

        


