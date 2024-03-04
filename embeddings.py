"""Descriptors under the Embedding designation."""

from numpy import ndarray
import matplotlib.pyplot as plt
from matplotlib import rcParams, rcParamsDefault
from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans
from persistence import RipsComplexDeepMD

class PBoW:
    def __init__(
        self, 
        rips_complex: RipsComplexDeepMD,
        hom_class: int = 1,
        nclusters:int = 5,
    ) -> None:
        self.rc_meta_ = rips_complex.meta_
        self.nclusters = nclusters
        self.diags_ = self._validate_diags(rips_complex.diag)
        self.descriptor = self._get_descriptor(self.diags_, hom=hom_class)
        self.features = self._get_features(self.descriptor)

    def _validate_diags(self, diags):
        if isinstance(diags[0], ndarray):
            raise ValueError("The input RipsComplex must be a super cell or list of persistence diagrams")

        self.super_cell_ = True
        self.hm_classes_ = list(range(len(diags[0])))
        return diags

    def _get_descriptor(self, diags, hom:int = 1):
        N = len(diags)

        curr_diags = [[d.tolist() for d in diag[hom]] for diag in diags]
        # get unique points for each pers diag
        convert_list = [v for cd in curr_diags for v in cd]
        unique_diags = [list(x) for x in set(tuple(x) for x in convert_list)]
        
        # collect clusters for each hom class
        kmeans = KMeans(n_clusters=self.nclusters, random_state=42)
        model = kmeans.fit(unique_diags)
        cluster_centers = model.cluster_centers_

        # find closest cluster index
        cluster_index = defaultdict(list)
        for cell, diag in enumerate(curr_diags):
            cluster_index[cell] = []
            for i, point in enumerate(diag):
                idx = np.sum((point - cluster_centers)**2, axis=1).argmin()
                cluster_index[cell].append((i, idx))

        # find cardinality
        vpbow_dict = {}
        for cell, indices in cluster_index.items():
            vpbow_dict[cell] = {i: 0 for i in range(self.nclusters)}
            for cluster_num in indices:
                vpbow_dict[cell][cluster_num] += 1
            vpbow_dict[cell] = list(vpbow_dict[cell].values())

        # convert to matrix 
        vpbow_matrix = np.zeros((N, self.nclusters))
        for cell in list(vpbow_dict.keys()):
            vpbow_matrix[cell, :] = vpbow_dict[cell]
        
        return vpbow_matrix

    def _get_features(
        self, descriptor: ndarray
    ) -> ndarray:
        features = descriptor.copy()
        features = np.sign(features) * np.sqrt(np.abs(features)) / np.linalg.norm(features, axis = 0)
        features = np.nan_to_num(features, copy=True, nan=0)
        return features 
    
    def plot(self) -> None:

        rcParams.update(rcParamsDefault)
        title = "Persistence Bag of Words"
        features = self.features
        num_hm_classes = features.shape[1]
        fig, ax = plt.subplots(1, num_hm_classes)
        plt.suptitle(title)
        for hm in range(num_hm_classes):
            im = ax[hm].imshow(features, aspect = 'auto')
            if hm == 0:
                ax[hm].set_ylabel("Cell Number")
            else:
                ax[hm].set_ylabel("")
                ax[hm].set_yticks([])
            ax[hm].set_xlabel("PBoW Vector")
            n_clusters = features.shape[2]
            if n_clusters <= 10:
                ax[hm].set_xticks(range(n_clusters))
                ax[hm].set_xticklabels(range(1, n_clusters+1))
            ax[hm].title.set_text(f"{hm}-Homology")
            if hm == num_hm_classes-1:
                plt.colorbar(im)
        plt.tight_layout()
        plt.show()
        return None