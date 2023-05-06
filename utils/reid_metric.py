# encoding: utf-8

import numpy as np
import torch
from ignite.metrics import Metric
from .reidtools import visualize_ranked_results
from data.datasets.eval_reid import eval_func
from .re_ranking import re_ranking


def cluster_matrix_show(matrix, pid):  #################显示ID特征聚类图
    from sklearn.manifold import TSNE
    import numpy as np
    import matplotlib.pyplot as plt
    import mpl_toolkits.mplot3d
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    # matrix = matrix.cpu().detach().numpy()
    matrix_tsne = tsne.fit_transform(matrix)
    # x_min, x_max = matrix.min(0),matrix.max(0)
    # x_norm = (matrix_tsne-x_min)/(x_max-x_min)
    # 定义数据
    x = [i[0] for i in matrix_tsne]
    y = [i[1] for i in matrix_tsne]
    # z = [i[2] for i in matrix_tsne]
    # ax = plt.subplot(projection='3d')  # 创建一个三维的绘图工程
    ax = plt.subplot()
    ax.set_title('Feature Cluster')
    scatter = ax.scatter(x, y, c=pid, cmap='rainbow')  # 绘制数据点
    ax.set_xlabel('X')  # 设置x坐标轴
    ax.set_ylabel('Y')  # 设置y坐标轴
    # ax.set_zlabel('Z')  # 设置z坐标轴
    plt.tight_layout()
    plt.legend(*scatter.legend_elements())
    plt.show()

class r1_mAP_mINP(Metric):
    def __init__(self, num_query, data_loader, max_rank=50, feat_norm='on', vis_rank='off'):
        super(r1_mAP_mINP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.vis_rank = vis_rank
        self.dataloader = data_loader

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))



    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'on':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        # query
        # self.num_query = 10#画t-sne图测试
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.cpu().numpy()
        cmc, mAP, mINP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
        if self.vis_rank == 'on':
            visualize_ranked_results(distmat, self.dataloader, self.num_query, 'image', save_dir='../vis_rank')
        # cluster_matrix_show(distmat, q_pids)
        return cmc, mAP, mINP


class r1_mAP_mINP_reranking(Metric):
    def __init__(self, num_query, data_loader, max_rank=50, feat_norm='on', vis_rank='on'):
        super(r1_mAP_mINP_reranking, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'on':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)

        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        # m, n = qf.shape[0], gf.shape[0]
        # distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        #           torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        # distmat.addmm_(1, -2, qf, gf.t())
        # distmat = distmat.cpu().numpy()
        print("Enter reranking")
        distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
        cmc, mAP, mINP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
        if self.vis_rank == 'on':
            visualize_ranked_results(distmat, self.dataloader, self.num_query, 'image', save_dir='../vis_rank')
        return cmc, mAP, mINP