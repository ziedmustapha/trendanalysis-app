import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import numpy as np
import logging
from ai import embed_text

def modified_dbscan(X, core_indices, eps, min_samples):
    """
    Implement a modified DBSCAN clustering algorithm.
    @param X - The dataset
    @param core_indices - Indices of core points in the dataset
    @param eps - The maximum distance between two samples for one to be considered as in the neighborhood of the other
    @param min_samples - The number of samples in a neighborhood for a point to be considered as a core point
    @return The cluster labels for each point in the dataset
    """
    labels = -np.ones(len(X), dtype=int)  # Initialize all points as noise with integer type
    cluster_id = 0

    for idx in core_indices:
        if labels[idx] != -1:  # Skip if already processed
            continue

        # Find neighbors
        neighbors = np.where(np.linalg.norm(X - X[idx], axis=1) < eps)[0]

        if len(neighbors) < min_samples:
            labels[idx] = -1  # Still noise, but this is redundant
        else:
            labels[idx] = cluster_id
            expand_cluster(X, labels, cluster_id, neighbors, eps, min_samples)
            cluster_id += 1

    return labels

def expand_cluster(X, labels, cluster_id, neighbors, eps, min_samples):
    """
    Expand the cluster by assigning cluster IDs to neighboring points.
    @param X - Data points
    @param labels - Cluster labels
    @param cluster_id - ID of the cluster being expanded
    @param neighbors - Neighboring points
    @param eps - Maximum distance between two samples for one to be considered as in the neighborhood of the other
    @param min_samples - The number of samples in a neighborhood for a point to be considered as a core point
    """
    i = 0
    while i < len(neighbors):
        neighbor_idx = neighbors[i]

        if labels[neighbor_idx] == -1:  # Noise
            labels[neighbor_idx] = cluster_id

        elif labels[neighbor_idx] == -2:  # Unclassified
            labels[neighbor_idx] = cluster_id
            new_neighbors = np.where(np.linalg.norm(X - X[neighbor_idx], axis=1) < eps)[0]
            if len(new_neighbors) >= min_samples:
                neighbors = np.append(neighbors, new_neighbors)  # Append new neighbors

        i += 1

def find_closest_papers(query_papers, all_papers, initial_eps=11, min_samples=0, top_k=8, similarity_threshold=0.5):
    """
    Find the closest older papers to each new query paper using SPECTER cosine similarity.
    """
    if not query_papers or not all_papers:
        return []

    if len(query_papers) > 10:
        query_papers = query_papers[:10]

    query_items = []
    for paper in query_papers:
        embedded_text = embed_text(paper.get("abstract"))
        if embedded_text is not None:
            query_items.append((paper, embedded_text))

    if not query_items:
        return []

    corpus_items = []
    for paper in all_papers:
        embedded_text = embed_text(paper.get("abstract"))
        if embedded_text is not None:
            corpus_items.append((paper, embedded_text))

    if not corpus_items:
        return []

    query_vecs = torch.stack([item[1] for item in query_items])
    all_paper_vecs = torch.stack([item[1] for item in corpus_items])
    similarities = cosine_similarity(query_vecs.numpy(), all_paper_vecs.numpy())

    final_results = []
    for i, (query_paper, _) in enumerate(query_items):
        ranked = similarities[i].argsort()[::-1]
        closest_papers = []
        for index in ranked:
            score = float(similarities[i][index])
            if score < similarity_threshold:
                continue
            paper = corpus_items[index][0]
            closest_papers.append((paper["title"], paper["abstract"], score))
            if len(closest_papers) >= top_k:
                break
        query_info = (query_paper["title"], query_paper["abstract"])
        final_results.append((query_info, closest_papers))

    return final_results

def fuse_similar_clusters(final_results, similarity_threshold=0.97):
    """
    Fuse similar clusters based on a similarity threshold.
    @param final_results - the final results to be clustered
    @param similarity_threshold - the threshold for considering clusters as similar (default is 0.8)
    """
    # Dictionary to store average embeddings of each cluster
    cluster_embeddings = {}
    
    # Calculate average embeddings for each cluster
    for (query_title, query_abstract), papers in final_results:
        embeddings = [embed_text(paper[1]) for paper in papers]
        if embeddings:
            avg_embedding = torch.mean(torch.stack(embeddings), dim=0)
            cluster_embeddings[query_title] = avg_embedding

    # Create titles list and embeddings matrix
    titles = list(cluster_embeddings.keys())
    if not titles:
        return []
    embeddings_matrix = torch.stack(list(cluster_embeddings.values())).numpy()
    
    # Compute similarity matrix
    similarity_matrix = cosine_similarity(embeddings_matrix)

    size_by_title = {query_title: len(papers) for (query_title, _query_abstract), papers in final_results}

    # Identify clusters to merge based on similarity threshold
    to_merge = {}
    for i in range(len(titles)):
        for j in range(i + 1, len(titles)):
            if similarity_matrix[i][j] > similarity_threshold:
                cluster_i_size = size_by_title.get(titles[i], 0)
                cluster_j_size = size_by_title.get(titles[j], 0)
                if cluster_i_size + cluster_j_size <= 40:
                    smaller, larger = sorted([i, j], key=lambda x: -size_by_title.get(titles[x], 0))
                    to_merge[titles[smaller]] = titles[larger]

    # Dictionary to store merged clusters
    merged_clusters = {}
    representative = {}
    for (query_title, query_abstract), papers in final_results:
        root_title = to_merge.get(query_title, query_title)
        if root_title not in merged_clusters:
            merged_clusters[root_title] = [[query_title], (query_title, query_abstract), []]
            # Merge chains mean the stored representative is not always root_title.
            representative[root_title] = query_title
        else:
            merged_clusters[root_title][0].append(query_title)
        merged_clusters[root_title][2].extend(papers)
        # Keep every absorbed query paper's abstract, otherwise it drops out of the PCA.
        if query_title != representative[root_title]:
            merged_clusters[root_title][2].append((query_title, query_abstract, 1.0)) # Adding a default similarity score

    # Convert merged_clusters dictionary to list
    new_final_results = list(merged_clusters.values())

    # Sort papers within each cluster by the third element (similarity score) in descending order
    for cluster in new_final_results:
        cluster[2].sort(key=lambda x: x[2], reverse=True)

    # Convert similarity scores to float for consistency
    for cluster in new_final_results:
        cluster[2] = [(title, abstract, float(similarity)) for (title, abstract, similarity) in cluster[2]]

    logging.info("Formed %s clusters", len(new_final_results))
    return new_final_results

def perform_pca(clusters):
    """
    Project new (query) papers and their related older papers into one shared 3D PCA space.

    Both groups must be fitted together, otherwise the new papers have no coordinates
    of their own and collapse onto the origin.
    @param clusters - list of clusters containing papers
    @return PCA results for query papers and for related papers
    """
    abstract_by_title = {}
    query_titles = []
    related_titles = []

    for cluster in clusters:
        root_title, root_abstract = cluster[1]
        abstract_by_title.setdefault(root_title, root_abstract)
        query_titles.extend(cluster[0])
        for title, abstract, _score in cluster[2]:
            abstract_by_title.setdefault(title, abstract)
            related_titles.append(title)

    ordered_titles = [
        title for title in dict.fromkeys(query_titles + related_titles)
        if title in abstract_by_title
    ]
    if not ordered_titles:
        return [], []

    embeddings = [embed_text(abstract_by_title[title]) for title in ordered_titles]
    embeddings_matrix = torch.stack(embeddings).numpy().astype(np.float64)

    n_components = min(3, embeddings_matrix.shape[0], embeddings_matrix.shape[1])
    if n_components < 1:
        return [], []

    pca_results = PCA(n_components=n_components).fit_transform(embeddings_matrix)

    coords_by_title = {}
    for idx, title in enumerate(ordered_titles):
        coords = list(pca_results[idx])
        while len(coords) < 3:
            coords.append(0.0)
        coords_by_title[title] = [float(value) for value in coords]

    pca_query_papers = [
        {'title': title, 'coords': coords_by_title[title]}
        for title in dict.fromkeys(query_titles)
        if title in coords_by_title
    ]
    pca_papers = [
        {'title': title, 'coords': coords_by_title[title]}
        for title in dict.fromkeys(related_titles)
        if title in coords_by_title
    ]

    logging.info(
        "PCA projected %s new + %s related papers", len(pca_query_papers), len(pca_papers)
    )
    return pca_query_papers, pca_papers