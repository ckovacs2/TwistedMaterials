"""Descriptors under the Embedding designation."""

from numpy import ndarray
import matplotlib.pyplot as plt
from matplotlib import rcParams, rcParamsDefault
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

class PBoW:
    def __init__(
        self, 
        diags: np.array,
        nclusters:int = 5,
    ) -> None:
        self.nclusters = nclusters
        self.cluster_centers = self._get_embeddings(diags)

    def _get_embeddings(self, diags):
        # get unique points for each pers diag
        unique_diags = [list(x) for x in set(tuple(x) for x in diags.tolist())]
        
        # collect clusters for each hom class
        kmeans = KMeans(n_clusters=self.nclusters, random_state=42)
        model = kmeans.fit(unique_diags)
        cluster_centers = model.cluster_centers_

        return cluster_centers

    def _get_descriptor(self, curr_diags):

        # find closest cluster index
        cluster_indices = []
        for (path, trans), curr_diag in curr_diags.items():
            for diag in curr_diag:
                cluster_idx = np.sum((diag - self.cluster_centers)**2, axis=1).argmin()
                cluster_indices.append((path, trans, cluster_idx))

        cluster_indices = pd.DataFrame(cluster_indices, 
                                    columns = ['path', 'trans', 'cluster_idx'])

        # find cardinality
        value_counts = cluster_indices.groupby(['path', 'trans', 'cluster_idx'], as_index=False).size()
        pivot_table = value_counts.pivot_table(index=['path', 'trans'], columns='cluster_idx', fill_value=0).reset_index(drop = True)
        pivot_table.columns = pivot_table.columns.droplevel(0)
        pivot_table.columns.name = None
        unmatched_cluster = list(set(range(self.nclusters)) - set(pivot_table.columns))
        if len(unmatched_cluster) != 0:
            for uc in unmatched_cluster:
                pivot_table[uc] = 0.
            pivot_table = pivot_table[list(range(self.nclusters))]
        return pivot_table

    def features(
        self, curr_diags
    ) -> ndarray:
        self.descriptor = self._get_descriptor(curr_diags=curr_diags)
        features = self.descriptor.copy()
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