"""Theme detection through clustering."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


def cluster_texts(embeddings, n_clusters: int = 3) -> list[int]:
    """Cluster embeddings, adapting the cluster count for small inputs."""

    if len(embeddings) == 0:
        return []

    matrix = np.asarray(embeddings)
    cluster_count = max(1, min(n_clusters, len(matrix)))

    kmeans = KMeans(n_clusters=cluster_count, n_init="auto", random_state=42)
    return kmeans.fit_predict(matrix).tolist()
