from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def pca_kmeans(features, n_components=10, n_clusters=3):
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(reduced)

    return reduced, labels
