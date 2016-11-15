from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask
import json
import numpy as np


cocoGt = COCO('./VOC2012-groundtruth.json')
cocoDt = cocoGt.loadRes('./VOC2012-predictions-example.json')

imgIds = sorted(cocoGt.getImgIds())

# running evaluation
cocoEval = COCOeval(cocoGt,cocoDt)
cocoEval.params.imgIds = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

categories = [line.strip() for line in open('categories.txt')]

for i in xrange(20):
  print 'AP@[0.5:0.95] for %s: %f' % (categories[i], np.mean(cocoEval.eval['precision'][:, :, i]))

for i in xrange(20):
  print 'AP@0.5 for %s: %f' % (categories[i], np.mean(cocoEval.eval['precision'][0, :, i]))

for i in xrange(20):
  print 'AP@0.7 for %s: %f' % (categories[i], np.mean(cocoEval.eval['precision'][4, :, i]))