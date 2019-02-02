#!/usr/bin/env python3
"""Thomas Liao Oct. Gen triplet net embedding for jin - on top of jin's code base"""

from __future__ import division, print_function, absolute_import

import logging
# import multiprocessing as mp
import os
import pickle
import cv2
import numpy as np
import shutil
from detection import detection_instances
from utils import data_utils
from utils import misc_utils
from utils.config import config
from triplet_reid import embed_class_tf



misc_utils.setup_logging()
base_dir = os.path.join(config['DET_BASE_DIR'], 'train/20180504a_resized')

dets_fmt = os.path.join(base_dir, 'seqs_triplet_embedding/{}.pkl')

out_fmt = os.path.join(base_dir, 'seqs_triplet_embedding/{}.pkl')

# modified input/output and wrapped in class for ease of use - tf version
tri_emb = embed_class_tf.TrinetEmbeddingTf()
tri_emb._initialize_static_graph()
tri_emb._restore_pretrained_model()

def process_seq(seq):
  # Load detections.
  logging.info('Loading detection instances...')
  dets_path = dets_fmt.format(seq)
  with open(dets_path, 'rb') as fin:
    dets = pickle.load(fin)

  for i, det in enumerate(dets):
    # Print progress.
    if (i + 1) % 100 == 0:
      logging.info('{}: processing detection {} of {}...'.format(
          seq, i + 1, len(dets)))

    # Skip low confidence detections.
    if det['score'] < 0.5:
      continue
    # Skip non-person detections.
    if det['object_category'] != 'Person':
      continue

    # Load image.
    img_path = data_utils.get_image_path(det['sequence'], det['frame'])
    temp = img_path.split('/')
    img = cv2.imread(img_path)
    # Crop image.
    x0, y0, x1, y1 = det['bbox'] 
    crop = img[int(y0):int(y1), int(x0):int(x1)]

    # Calculate embedding.
    # TODO:
    crop_input = cv2.resize(crop, (128, 256))
    crop_input = crop_input[np.newaxis,]
    embedding = tri_emb.gen_embedding(crop_input)
    embedding = embedding[0].tolist()
    det['triplet_embedding'] = embedding



  # Write output.
  out_path = out_fmt.format(seq)
  with open(out_path, 'wb') as fout:
    pickle.dump(dets, fout)

if __name__ == '__main__':

  for seq in data_utils.train_seqs:
    process_seq(seq)

  #






  # source_dir = '/data/diva/annotations_cache/seqs'
  # target_dir = '/work_12t/tliao4/tracking/diva_results/gt/train/dummy_file/seqs_gt'
  # for seq in data_utils.train_seqs:
  #     from_path = os.path.join(source_dir, seq+'.pkl')
  #     to_path = os.path.join(target_dir, seq+'.pkl')
  #     shutil.copyfile(from_path, to_path)




