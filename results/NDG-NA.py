import numpy as np
from scipy.ndimage import uniform_filter
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.csgraph import shortest_path

def NDG_01(W_in):
    """Community detection based network denoising"""
    N = W_in.shape[0]
    denoised_adj = W_in.copy()
    
    num_iterations = 20
    threshold = 0.5
    
    for _ in range(num_iterations):
        degrees = np.sum(denoised_adj, axis=1)
        modularity_matrix = denoised_adj - np.outer(degrees, degrees) / np.sum(degrees)
        
        eigenvalues, eigenvectors = np.linalg.eig(modularity_matrix)
        
        community_vector = eigenvectors[:, np.argmax(eigenvalues)]
        communities = np.where(community_vector > np.mean(community_vector), 1, 0)
        
        for i in range(N):
            for j in range(N):
                if communities[i] != communities[j]:
                    denoised_adj[i, j] += 0.2
                else:
                    denoised_adj[i, j] *= 0.7
        
        denoised_adj = (denoised_adj + denoised_adj.T) / 2
        denoised_adj = np.maximum(denoised_adj, 0)
        denoised_adj /= np.max(denoised_adj)
    
    return denoised_adj

def NDG_02(W_in):
    """Three-step network denoising mechanism"""
    W = np.array(W_in)
    W = (W + W.T) / 2
    
    W_norm = W / np.max(W) if np.max(W) > 0 else W
    
    direct_strength = np.where(W_norm > 0, W_norm, 0)
    
    indirect_strength = np.dot(direct_strength, direct_strength)
    
    threshold = np.percentile(direct_strength[direct_strength > 0], 90)
    
    W_enhanced = np.copy(W_norm)
    for i in range(W_enhanced.shape[0]):
        for j in range(W_enhanced.shape[1]):
            score = 1.5 * direct_strength[i][j] + 0.3 * indirect_strength[i][j]
            if score > threshold:
                W_enhanced[i][j] = min(W_enhanced[i][j] + 0.25, 1)
            elif score < threshold / 2:
                W_enhanced[i][j] = max(W_enhanced[i][j] - 0.1, 0)
    
    W_enhanced = (W_enhanced - W_enhanced.min()) / (W_enhanced.max() - W_enhanced.min())
    np.fill_diagonal(W_enhanced, 0)
    
    return W_enhanced

def NDG_03(W_in):
    """Community detection based network denoising algorithm"""
    N = W_in.shape[0]
    denoised_adj = W_in.copy()
    
    num_iterations = 50
    
    for _ in range(num_iterations):
        degrees = np.sum(denoised_adj, axis=1)
        modularity_matrix = denoised_adj - np.outer(degrees, degrees) / np.sum(degrees)

        eigenvalues, eigenvectors = np.linalg.eig(modularity_matrix)
        
        community_vector = eigenvectors[:, np.argmax(eigenvalues)]
        communities = np.where(community_vector > np.mean(community_vector), 1, 0)
        
        for i in range(N):
            for j in range(N):
                if communities[i] != communities[j]:
                    denoised_adj[i, j] += 0.2
                else:
                    denoised_adj[i, j] *= 0.7

    return denoised_adj

def NDG_04(W_in):
    """Global structure similarity based network denoising"""
    W = np.array(W_in)
    W = (W + W.T) / 2
    
    for _ in range(5):
        sim_matrix = cosine_similarity(W)
        
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                if i != j:
                    if sim_matrix[i, j] > 0.5:
                        W[i, j] = 0.6 * W[i, j] + 0.4 * sim_matrix[i, j]
                    else:
                        W[i, j] = 0.4 * W[i, j] + 0.6 * sim_matrix[i, j]
        
        W = (W + W.T) / 2
    
    np.fill_diagonal(W, 0)
    
    return W

def NDG_05(W_in):
    """Uniform averaging based network denoising"""
    denoised_adj = np.copy(W_in)
    num_iterations = 10
    kernel_size = 3

    for _ in range(num_iterations):
        smoothed_adj = uniform_filter(denoised_adj, size=kernel_size)
        smoothed_adj[smoothed_adj < 0.05 * np.max(smoothed_adj)] = 0
        row_sums = smoothed_adj.sum(axis=1, keepdims=True)
        denoised_adj = smoothed_adj / (row_sums + 1e-10)
        denoised_adj = (denoised_adj + denoised_adj.T) / 2
        np.fill_diagonal(denoised_adj, 0)

    return denoised_adj

def NDG_06(W_in):
    """Community detection based network denoising algorithm"""
    W = np.array(W_in)
    W = (W + W.T) / 2
    
    degree = np.sum(W, axis=1)
    total_edges = np.sum(degree)
    modularity_matrix = W - np.outer(degree, degree) / total_edges
    
    eigenvalues, eigenvectors = np.linalg.eigh(modularity_matrix)
    
    significant_eigenvector = eigenvectors[:, np.argmax(eigenvalues)]
    
    normalized_strength = (significant_eigenvector - np.min(significant_eigenvector)) / (
            np.max(significant_eigenvector) - np.min(significant_eigenvector))
    threshold = np.percentile(normalized_strength, 80)
    
    W_enhanced = np.copy(W)
    for i in range(W_enhanced.shape[0]):
        for j in range(W_enhanced.shape[1]):
            if normalized_strength[i] > threshold and normalized_strength[j] > threshold:
                W_enhanced[i][j] = min(W_enhanced[i][j] + 0.2, 1)
            else:
                W_enhanced[i][j] = max(W_enhanced[i][j] - 0.05, 0)
    
    W_enhanced = (W_enhanced - W_enhanced.min()) / (W_enhanced.max() - W_enhanced.min())
    np.fill_diagonal(W_enhanced, 0)
    
    return W_enhanced

def NDG_07(W_in):
    """Node centrality and community influence based network denoising"""
    W = np.array(W_in)
    W = (W + W.T) / 2
    
    dist_matrix = shortest_path(W, directed=False, unweighted=False)
    
    centrality = np.sum(W, axis=1)
    centrality = centrality / np.max(centrality)
    
    community_influence = np.dot(W, W.T) / np.sum(W)
    
    decay_factor = 0.5
    enhanced_adj = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            if i == j:
                enhanced_adj[i, j] = 0
            else:
                enhanced_adj[i, j] = W[i, j] * community_influence[i, j] * np.exp(-decay_factor * dist_matrix[i, j] * (1 - centrality[i] * centrality[j]))
    
    enhanced_adj = enhanced_adj / np.max(enhanced_adj)
    np.fill_diagonal(enhanced_adj, 0)
    
    return enhanced_adj