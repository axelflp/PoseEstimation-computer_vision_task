import numpy as np
from itertools import chain
import os

from skimage import io, transform
import matplotlib.pyplot as plt
import matplotlib.collections as mc
import matplotlib.image as mpimg


# function to plot

# edges between points, each value is th index of a point, there are 15 edges
connections = [[0,2],[2,1],[3,4],[4,5],[5,6],[6,7],[4,8],[8,9],[9,10],[4,11],[11,12],[12,13],[11,14],[14,15],[11,16]]

def brid_pl(landmarks):

    landmarks = np.reshape(landmarks,(-1,2)).tolist()
    # list of pair of points that are linked
    connec = [[landmarks[j],landmarks[k]] for j, k in connections]
    # [[x,y],[x, y]] -> [[x,x],[y,y]]
    connec_pl = chain(*np.array(connec).transpose((0,2,1)).tolist())
    
    return np.array(connec).astype('float'), connec_pl


def plot_image_versions(data):
    bbox = data['bbox']
    x1, y1 = int(bbox[0]), int(bbox[1])
    x2, y2 = int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])

    original_image = mpimg.imread(data['file'])
    raw_image = original_image[y1:y2, x1:x2]
    raw_landmarks = np.array(data['landmarks'])

    # Rescaling the data
    new_h, new_w = 200, 200
    rescaled_image = transform.resize(raw_image, (new_h, new_w))
    rescaled_landmarks = (
        raw_landmarks.reshape(-1, 2) *
        [new_w/(x2-x1), new_h/(y2-y1)]
    ).flatten()

    # Original image
    h, w = original_image.shape[:2]
    ratio = h / w

    fig = plt.figure(figsize=(14, 14 * ratio + 6), constrained_layout=False)
    subfigs = fig.subfigures(3, 1, height_ratios=[ratio * 1.5, 1, 1], hspace=0.0)

    # Pricipal plot
    subfigs[0].subplots_adjust(top=0.85, bottom=0.00)
    subfigs[0].suptitle("Original Image", fontsize=20, fontweight='heavy', style='oblique', y=0.94)
    ax0 = subfigs[0].subplots(1, 1)
    ax0.imshow(original_image)
    ax0.axis("off")
    

    def plot_pair(subfig, image, landmarks, title):
        subfig.subplots_adjust(top=0.85, bottom=0.05)
        subfig.suptitle(title, fontsize=20, fontweight='heavy', style='oblique', y=0.94)
        ax_left, ax_right = subfig.subplots(1, 2, gridspec_kw={'wspace': -0.2})
        

        x, y = landmarks.reshape(-1, 2).T
        br, br_plot = brid_pl(landmarks)

        # right image
        ax_right.imshow(image)
        ax_right.scatter(x, y, s=40, c='red')
        ax_right.plot(*br_plot, color="royalblue", linewidth=2)
        ax_right.set_aspect("equal")

        # left iamge
        lc = mc.LineCollection(br, linewidths=3)
        ax_left.add_collection(lc)
        ax_left.scatter(x, y, s=60, c='red')
        ax_left.set_aspect("equal")


        ax_left.set_xlim(ax_right.get_xlim())
        ax_left.set_ylim(ax_right.get_ylim())
        
    plot_pair(subfigs[1], raw_image, raw_landmarks, "Cropped image")
    plot_pair(subfigs[2], rescaled_image, rescaled_landmarks, "Rescaled image")

    # fig.savefig(f"{idx}.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_predictions(data1, data2, landmarks_pred):

    def plot_pair(subfig, image, landmarks, landmarks_pred):
        ax_left, ax_right = subfig.subplots(1, 2, gridspec_kw={'wspace': -0.01})
        

        x, y = landmarks.reshape(-1, 2).T
        br, br_plot = brid_pl(landmarks)

        # left image
        ax_left.imshow(image)
        ax_left.scatter(x, y, s=40, c='red')
        ax_left.plot(*br_plot, color="royalblue", linewidth=2)
        ax_left.set_aspect("equal")
        ax_left.set_title('Target', fontsize=20, fontweight='heavy', style='oblique')

        x, y = landmarks_pred.reshape(-1, 2).T
        br, br_plot = brid_pl(landmarks_pred)
        
        # right iamge
        ax_right.imshow(image)
        ax_right.scatter(x, y, s=40, c='red')
        ax_right.plot(*br_plot, color="royalblue", linewidth=2)
        ax_right.set_aspect("equal")
        ax_right.set_title('Prediction', fontsize=20, fontweight='heavy', style='oblique')

    fig = plt.figure(figsize=(16, 12), constrained_layout=False)
    subfigs = fig.subfigures(2, 1)
    
    for idx, dat in enumerate([data1, data2]):
        bbox = dat['bbox']
        x1, y1 = int(bbox[0]), int(bbox[1])
        x2, y2 = int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
    
        original_image = mpimg.imread(dat['file'])
        raw_image = original_image[y1:y2, x1:x2]
        raw_landmarks = np.array(dat['landmarks'])
        raw_landmarks_pred = landmarks_pred[idx]
    
        # rescaling
        new_h, new_w = 200, 200
        rescaled_image = transform.resize(raw_image, (new_h, new_w))
        rescaled_landmarks = (
            raw_landmarks.reshape(-1, 2) *
            [new_w/(x2-x1), new_h/(y2-y1)]
        ).flatten()

        plot_pair(subfigs[idx], rescaled_image, rescaled_landmarks, raw_landmarks_pred)

    subfigs[1].subplots_adjust(top=0.95, bottom=0.05)
    # fig.savefig(f"res.png", dpi=300, bbox_inches='tight')