__author__ = 'tsungyi'

import numpy as np
import datetime
import time
from collections import defaultdict
import mask
import copy

class COCOeval:
    # Interface for evaluating detection on the Microsoft COCO dataset.
    #
    # The usage for CocoEval is as follows:
    #  cocoGt=..., cocoDt=...       # load dataset and results
    #  E = CocoEval(cocoGt,cocoDt); # initialize CocoEval object
    #  E.params.recThrs = ...;      # set parameters as desired
    #  E.evaluate();                # run per image evaluation
    #  E.accumulate();              # accumulate per image results
    #  E.summarize();               # display summary metrics of results
    # For example usage see evalDemo.m and http://mscoco.org/.
    #
    # The evaluation parameters are as follows (defaults in brackets):
    #  imgIds     - [all] N img ids to use for evaluation
    #  catIds     - [all] K cat ids to use for evaluation
    #  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
    #  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
    #
    # evaluate(): evaluates detections on every image and every category and
    # concats the results into the "evalImgs" with fields:
    #  dtIds      - [1xD] id for each of the D detections (dt)
    #  gtIds      - [1xG] id for each of the G ground truths (gt)
    #  dtMatches  - [TxD] matching gt id at each IoU or 0
    #  gtMatches  - [TxG] matching dt id at each IoU or 0
    #  dtScores   - [1xD] confidence of each dt
    #  gtIgnore   - [1xG] ignore flag for each gt
    #  dtIgnore   - [TxD] ignore flag for each dt at each IoU
    #
    # accumulate(): accumulates the per-image, per-category evaluation
    # results in "evalImgs" into the dictionary "eval" with fields:
    #  params     - parameters used for evaluation
    #  date       - date evaluation was performed
    #  counts     - [T,R,K,A,M] parameter dimensions (see above)
    #  precision  - [TxRxKxAxM] precision for every evaluation setting
    #  recall     - [TxKxAxM] max recall for every evaluation setting
    # Note: precision and recall==-1 for settings with no gt objects.
    #
    # See also coco, mask, pycocoDemo, pycocoEvalDemo
    #
    # Microsoft COCO Toolbox.      version 2.0
    # Data, paper, and tutorials available at:  http://mscoco.org/
    # Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
    # Licensed under the Simplified BSD License [see coco/license.txt]
    def __init__(self, cocoGt=None, cocoDt=None):
        '''
        Initialize CocoEval using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None
        '''
        self.cocoGt   = cocoGt              # ground truth COCO API
        self.cocoDt   = cocoDt              # detections COCO API
        self.params   = {}                  # evaluation parameters
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results [KxAxI] elements
        self.eval     = {}                  # accumulated evaluation results
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        self.params = Params() # parameters
        self._paramsEval = {}               # parameters for evaluation
        self.stats = []                     # result summarization
        self.ious = {}                      # ious between all gts and dts
        if not cocoGt is None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())


    def _prepare(self):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        '''
        def _toMask(objs, coco):
            # modify segmentation by reference
            for obj in objs:
                t = coco.imgs[obj['image_id']]
                if type(obj['segmentation']) == list:
                    if type(obj['segmentation'][0]) == dict:
                        print 'debug'
                    obj['segmentation'] = mask.frPyObjects(obj['segmentation'],t['height'],t['width'])
                    if len(obj['segmentation']) == 1:
                        obj['segmentation'] = obj['segmentation'][0]
                    else:
                        # an object can have multiple polygon regions
                        # merge them into one RLE mask
                        obj['segmentation'] = mask.merge(obj['segmentation'])
                elif type(obj['segmentation']) == dict and type(obj['segmentation']['counts']) == list:
                    obj['segmentation'] = mask.frPyObjects([obj['segmentation']],t['height'],t['width'])[0]
                elif type(obj['segmentation']) == dict and \
                    (type(obj['segmentation']['counts']) == unicode or type(obj['segmentation']['counts']) == str):
                    pass
                else:
                    raise Exception('segmentation format not supported.')
        p = self.params
        gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))

        # convert ground truth to mask
        _toMask(gts, self.cocoGt)
        _toMask(dts, self.cocoDt)
        # set ignore flag
        for gt in gts:
            gt["ignore"] = 0
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        for gt in gts:
            self._gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results
        self.eval     = {}                  # accumulated evaluation results

    def evaluate(self):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        print 'Running per image evaluation...      '
        p = self.params
        p.imgIds = list(np.unique(p.imgIds))
        p.catIds = list(np.unique(p.catIds))
        self.params=p

        self._prepare()
        # loop through images, area range, max detection number
        catIds = p.catIds

        computeIoU = self.computeIoU
        self.ious = {(imgId, catId): computeIoU(imgId, catId) \
                        for imgId in p.imgIds
                        for catId in catIds}

        evaluateImg = self.evaluateImg
        self.evalImgs = [evaluateImg(imgId, catId)
                 for catId in catIds
                 for imgId in p.imgIds
             ]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print 'DONE (t=%0.2fs).'%(toc-tic)

    def computeIoU(self, imgId, catId):
        p = self.params
        gt = self._gts[imgId,catId]
        dt = self._dts[imgId,catId]
        if len(gt) == 0 and len(dt) ==0:
            return []
        inds = np.argsort(map(lambda x:-x["score"], dt),kind='mergesort')
        dt = [dt[i] for i in inds]

        g = [g['segmentation'] for g in gt]
        d = [d['segmentation'] for d in dt]
        # compute iou between each dt and gt region
        iscrowd = [0 for o in gt]
        ious = mask.iou(d,g,iscrowd)
        return ious

    def evaluateImg(self, imgId, catId):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        p = self.params
        gt = self._gts[imgId,catId]
        dt = self._dts[imgId,catId]
        if len(gt) == 0 and len(dt) ==0:
            return None

        for g in gt:
            if g['ignore']:
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind="mergesort")
        gt = map(lambda i: gt[i], gtind)
        dtind = np.argsort([-d['score'] for d in dt], kind="mergesort")
        dt = map(lambda i: dt[i], dtind)
        iscrowd = [0 for o in gt]
        # load computed ious
        ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm  = np.zeros((T,G))
        dtm  = np.zeros((T,D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T,D))
        if not len(ious)==0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t,1-1e-10])
                    m   = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind,gind]>0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m>-1 and gtIg[m]==0 and gtIg[gind]==1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind,gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou=ious[dind,gind]
                        m=gind
                    # if match made store id of match for both dt and gt
                    if m ==-1:
                        continue
                    dtIg[tind,dind] = gtIg[m]
                    dtm[tind,dind]  = gt[m]['id']
                    gtm[tind,m]     = d['id']
        # store results for given image and category
        return {
                'image_id':     imgId,
                'category_id':  catId,
                'dtIds':        [d['id'] for d in dt],
                'gtIds':        [g['id'] for g in gt],
                'dtMatches':    dtm,
                'gtMatches':    gtm,
                'dtScores':     [d['score'] for d in dt],
                'gtIgnore':     gtIg,
                'dtIgnore':     dtIg,
            }

    def accumulate(self, p = None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print 'Accumulating evaluation results...   '
        tic = time.time()
        if not self.evalImgs:
            print 'Please run evaluate() first'
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds
        T           = len(p.iouThrs)
        R           = len(p.recThrs)
        K           = len(p.catIds)
        precision   = -np.ones((T,R,K)) # -1 for the precision of absent categories
        recall      = -np.ones((T,K))

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds
        setK = set(catIds)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds)  if k in setK]
        i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]
        I0 = len(_pe.imgIds)
        # retrieve E at each category, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0*I0
            E = [self.evalImgs[Nk+i] for i in i_list]
            E = filter(None, E)
            if len(E) == 0:
                continue
            dtScores = np.concatenate([e['dtScores'] for e in E])

            # different sorting method generates slightly different results.
            # mergesort is used to be consistent as Matlab implementation.
            inds = np.argsort(-dtScores, kind='mergesort')

            dtm  = np.concatenate([e['dtMatches'] for e in E], axis=1)[:,inds]
            dtIg = np.concatenate([e['dtIgnore']  for e in E], axis=1)[:,inds]
            gtIg = np.concatenate([e['gtIgnore'] for e in E])
            npig = np.count_nonzero(gtIg==0 )
            if npig == 0:
                continue
            tps = np.logical_and(               dtm,  np.logical_not(dtIg) )
            fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )

            tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
            fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
            for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                tp = np.array(tp)
                fp = np.array(fp)
                nd = len(tp)
                rc = tp / npig
                pr = tp / (fp+tp+np.spacing(1))
                q  = np.zeros((R,))

                if nd:
                    recall[t,k] = rc[-1]
                else:
                    recall[t,k] = 0

                # numpy is slow without cython optimization for accessing elements
                # use python array gets significant speed improvement
                pr = pr.tolist(); q = q.tolist()

                for i in range(nd-1, 0, -1):
                    if pr[i] > pr[i-1]:
                        pr[i-1] = pr[i]

                inds = np.searchsorted(rc, p.recThrs, side='left')
                try:
                    for ri, pi in enumerate(inds):
                        q[ri] = pr[pi]
                except:
                    pass
                precision[t,:,k] = np.array(q)
        self.eval = {
            'params': p,
            'counts': [T, R, K],
            'date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'precision': precision,
            'recall':   recall,
        }
        toc = time.time()
        print 'DONE (t=%0.2fs).'%( toc-tic )

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize(iouThr=None):
            p = self.params
            iStr        = ' {:<18} @[ IoU={:<9}] = {}'
            titleStr    = 'Mean Average Precision'
            iouStr      = '%0.2f:%0.2f'%(p.iouThrs[0], p.iouThrs[-1]) if iouThr is None else '%0.2f'%(iouThr)

            # dimension of precision: [TxRxK]
            s = self.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:,:,:]

            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])
            print iStr.format(titleStr, iouStr, '%.3f'%(float(mean_s)))
            return mean_s

        def summarize():
            stats = np.zeros((12,))
            stats[0] = _summarize()
            stats[1] = _summarize(iouThr=.5)
            stats[2] = _summarize(iouThr=.7)
            return stats

        if not self.eval:
            raise Exception('Please run accumulate() first')

        self.stats = summarize()

    def __str__(self):
        self.summarize()

class Params:
    '''
    Params for coco evaluation api
    '''
    def setDetParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, np.round((1.00 - .0) / .01) + 1, endpoint=True)

    def __init__(self):
        self.setDetParams()
