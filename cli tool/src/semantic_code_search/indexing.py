# semantic_code_search/indexing.py

import numpy as np
import random
import heapq
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional, Union, Callable
import time

class IndexingStructure:
    """Base class for all advanced indexing structures."""
    
    def __init__(self, name: str):
        self.name = name
        self.build_time = 0
        
    def build(self, embeddings: np.ndarray, documents: List[Dict[str, Any]]) -> None:
        """Build the index structure from embeddings and document metadata."""
        start_time = time.time()
        self._build_implementation(embeddings, documents)
        self.build_time = time.time() - start_time
        print(f"Built {self.name} index in {self.build_time:.2f} seconds")
        
    def _build_implementation(self, embeddings: np.ndarray, documents: List[Dict[str, Any]]) -> None:
        """Implementation-specific building logic."""
        raise NotImplementedError
        
    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[float, int]]:
        """
        Search for k nearest neighbors to query_embedding.
        
        Returns:
        List of tuples (score, document_index)
        """
        raise NotImplementedError


# Red-Black Tree Node Implementation
class RBNode:
    """Red-Black Tree Node."""
    
    def __init__(self, key: float, value: int, is_red: bool = True):
        self.key = key  # Projection value or similarity score
        self.value = value  # Document index
        self.left = None
        self.right = None
        self.parent = None
        self.is_red = is_red  # True if red, False if black


class RedBlackTreeIndex(IndexingStructure):
    """Red-Black Tree implementation for similarity search."""
    
    def __init__(self):
        super().__init__("Red-Black Tree")
        self.root = None
        self.nil = RBNode(0, -1, False)  # Sentinel node
        self.nil.left = self.nil
        self.nil.right = self.nil
        self.nil.parent = self.nil
        self.root = self.nil
        self.embeddings = None
        self.documents = None
        self.projection_vectors = []
        self.num_projections = 3  # Number of random projections to use
        
    def _build_implementation(self, embeddings: np.ndarray, documents: List[Dict[str, Any]]) -> None:
        """Build Red-Black Tree index with multiple random projections."""
        self.embeddings = embeddings
        self.documents = documents
        
        # Create multiple random projection vectors
        self.projection_vectors = []
        for _ in range(self.num_projections):
            projection = np.random.randn(embeddings.shape[1])
            projection /= np.linalg.norm(projection)
            self.projection_vectors.append(projection)
            
        # Insert all documents into the tree using the first projection
        main_projection = self.projection_vectors[0]
        projected_values = embeddings @ main_projection
        
        for i, proj_val in enumerate(projected_values):
            self._insert(proj_val, i)
    
    def _left_rotate(self, x: RBNode) -> None:
        """Left rotation for Red-Black Tree balancing."""
        y = x.right
        x.right = y.left
        
        if y.left != self.nil:
            y.left.parent = x
            
        y.parent = x.parent
        
        if x.parent == self.nil:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
            
        y.left = x
        x.parent = y
    
    def _right_rotate(self, y: RBNode) -> None:
        """Right rotation for Red-Black Tree balancing."""
        x = y.left
        y.left = x.right
        
        if x.right != self.nil:
            x.right.parent = y
            
        x.parent = y.parent
        
        if y.parent == self.nil:
            self.root = x
        elif y == y.parent.left:
            y.parent.left = x
        else:
            y.parent.right = x
            
        x.right = y
        y.parent = x
    
    def _insert_fixup(self, z: RBNode) -> None:
        """Fix Red-Black Tree properties after insertion."""
        while z.parent.is_red:
            if z.parent == z.parent.parent.left:
                y = z.parent.parent.right
                
                if y.is_red:
                    z.parent.is_red = False
                    y.is_red = False
                    z.parent.parent.is_red = True
                    z = z.parent.parent
                else:
                    if z == z.parent.right:
                        z = z.parent
                        self._left_rotate(z)
                    
                    z.parent.is_red = False
                    z.parent.parent.is_red = True
                    self._right_rotate(z.parent.parent)
            else:
                y = z.parent.parent.left
                
                if y.is_red:
                    z.parent.is_red = False
                    y.is_red = False
                    z.parent.parent.is_red = True
                    z = z.parent.parent
                else:
                    if z == z.parent.left:
                        z = z.parent
                        self._right_rotate(z)
                    
                    z.parent.is_red = False
                    z.parent.parent.is_red = True
                    self._left_rotate(z.parent.parent)
                    
            if z == self.root:
                break
                
        self.root.is_red = False
    
    def _insert(self, key: float, value: int) -> None:
        """Insert a new node with key and value into the Red-Black Tree."""
        z = RBNode(key, value)
        y = self.nil
        x = self.root
        
        while x != self.nil:
            y = x
            if z.key < x.key:
                x = x.left
            else:
                x = x.right
                
        z.parent = y
        
        if y == self.nil:
            self.root = z
        elif z.key < y.key:
            y.left = z
        else:
            y.right = z
            
        z.left = self.nil
        z.right = self.nil
        z.is_red = True
        
        self._insert_fixup(z)
    
    def _search_tree(self, key: float, k: int = 10) -> List[int]:
        """Search the Red-Black Tree for nodes closest to the key."""
        # Find the closest node to the key
        closest_node = self._find_closest(self.root, key)
        if closest_node == self.nil:
            return []
            
        # Use a priority queue to find k nearest neighbors
        candidates = []
        self._inorder_nearest(self.root, key, k, candidates)
        
        # Extract the indices
        return [idx for _, idx in sorted(candidates, key=lambda x: abs(x[0] - key))[:k]]
    
    def _find_closest(self, node: RBNode, key: float) -> RBNode:
        """Find the node closest to the given key."""
        if node == self.nil:
            return self.nil
            
        if key == node.key:
            return node
            
        if key < node.key:
            if node.left == self.nil:
                return node
            
            left_closest = self._find_closest(node.left, key)
            if abs(left_closest.key - key) < abs(node.key - key):
                return left_closest
            return node
        else:
            if node.right == self.nil:
                return node
                
            right_closest = self._find_closest(node.right, key)
            if abs(right_closest.key - key) < abs(node.key - key):
                return right_closest
            return node
    
    def _inorder_nearest(self, node: RBNode, key: float, k: int, candidates: List[Tuple[float, int]]) -> None:
        """Traverse the tree in-order and find k nearest nodes to the key."""
        if node == self.nil:
            return
            
        self._inorder_nearest(node.left, key, k, candidates)
        
        # Use a bounded priority queue approach
        if len(candidates) < k:
            heapq.heappush(candidates, (-abs(node.key - key), node.value))
        else:
            # Replace if this node is closer than the furthest one in the queue
            if abs(node.key - key) < -candidates[0][0]:
                heapq.heappushpop(candidates, (-abs(node.key - key), node.value))
                
        self._inorder_nearest(node.right, key, k, candidates)
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[float, int]]:
        """Search for documents most similar to the query embedding."""
        # Project query to the same space as the indexed documents
        results = []
        
        # Use multiple projections and aggregate results
        for proj_vec in self.projection_vectors:
            query_projection = query_embedding @ proj_vec
            candidate_indices = self._search_tree(query_projection, k * 2)
            
            # Compute actual cosine similarities for the candidates
            for idx in candidate_indices:
                sim_score = np.dot(query_embedding, self.embeddings[idx]) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(self.embeddings[idx])
                )
                results.append((float(sim_score), idx))
        
        # Deduplicate and get top k
        seen = set()
        unique_results = []
        for score, idx in sorted(results, key=lambda x: x[0], reverse=True):
            if idx not in seen:
                seen.add(idx)
                unique_results.append((score, idx))
                if len(unique_results) >= k:
                    break
                    
        return unique_results


# Skip List implementation
class SkipNode:
    """Node in Skip List structure."""
    
    def __init__(self, key: float, value: int, level: int):
        self.key = key
        self.value = value
        self.forward = [None] * (level + 1)
        

class SkipListIndex(IndexingStructure):
    """Skip List implementation for similarity search."""
    
    def __init__(self, max_level: int = 16, p: float = 0.5):
        super().__init__("Skip List")
        self.max_level = max_level
        self.p = p
        self.level = 0
        self.header = SkipNode(-float('inf'), -1, max_level)
        self.embeddings = None
        self.documents = None
        self.projection_vector = None
    
    def _random_level(self) -> int:
        """Generate a random level for a new node with geometric distribution."""
        level = 0
        while random.random() < self.p and level < self.max_level:
            level += 1
        return level
    
    def _build_implementation(self, embeddings: np.ndarray, documents: List[Dict[str, Any]]) -> None:
        """Build Skip List index using a random projection."""
        self.embeddings = embeddings
        self.documents = documents
        
        # Create a random projection vector
        self.projection_vector = np.random.randn(embeddings.shape[1])
        self.projection_vector /= np.linalg.norm(self.projection_vector)
        
        # Project embeddings to 1D
        projected_values = embeddings @ self.projection_vector
        
        # Insert all documents into the skip list
        for i, proj_val in enumerate(projected_values):
            self._insert(proj_val, i)
    
    def _insert(self, key: float, value: int) -> None:
        """Insert a new node with key and value into the Skip List."""
        update = [None] * (self.max_level + 1)
        current = self.header
        
        # Find the position to insert
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
            update[i] = current
        
        # Create a new node with random level
        level = self._random_level()
        if level > self.level:
            for i in range(self.level + 1, level + 1):
                update[i] = self.header
            self.level = level
        
        new_node = SkipNode(key, value, level)
        
        # Update pointers
        for i in range(level + 1):
            new_node.forward[i] = update[i].forward[i]
            update[i].forward[i] = new_node
    
    def _search(self, key: float) -> Optional[SkipNode]:
        """Search for a node with the given key."""
        current = self.header
        
        # Traverse the skip list
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
        
        # Move to the next node
        current = current.forward[0]
        
        # Return the node if found
        if current and current.key == key:
            return current
        return None
    
    def _find_nearest(self, key: float, k: int) -> List[int]:
        """Find k nearest nodes to the given key."""
        current = self.header
        
        # Find the position closest to the key
        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].key < key:
                current = current.forward[i]
        
        # Get the successor node
        current = current.forward[0]
        
        candidates = []
        # Collect nodes before and after the key
        before = []
        after = []
        
        # Collect nodes after the key
        node = current
        while node and len(after) < k:
            after.append((node.key, node.value))
            node = node.forward[0]
        
        # Collect nodes before the key
        node = self.header
        prev = None
        while node != current:
            prev = node
            node = node.forward[0]
            if prev != self.header:
                before.append((prev.key, prev.value))
        
        # Sort before nodes by distance to key (closest last)
        before.sort(key=lambda x: abs(x[0] - key), reverse=True)
        before = before[:k]
        before.reverse()  # Now closest first
        
        # Collect k nearest nodes
        i, j = 0, 0
        while len(candidates) < k and (i < len(before) or j < len(after)):
            if i < len(before) and (j >= len(after) or abs(before[i][0] - key) <= abs(after[j][0] - key)):
                candidates.append(before[i][1])
                i += 1
            elif j < len(after):
                candidates.append(after[j][1])
                j += 1
        
        return candidates
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[float, int]]:
        """Search for documents most similar to the query embedding."""
        # Project query to the same space as the indexed documents
        query_projection = query_embedding @ self.projection_vector
        
        # Find nearest candidates
        candidate_indices = self._find_nearest(query_projection, k * 2)
        
        # Compute actual cosine similarities for the candidates
        results = []
        for idx in candidate_indices:
            sim_score = np.dot(query_embedding, self.embeddings[idx]) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(self.embeddings[idx])
            )
            results.append((float(sim_score), idx))
        
        # Return top k
        return sorted(results, key=lambda x: x[0], reverse=True)[:k]


# Binary Max Heap implementation
class BinaryMaxHeap:
    """Binary Max Heap for priority queue operations."""
    
    def __init__(self):
        self.heap = []
        
    def parent(self, i: int) -> int:
        """Get parent index"""
        return (i - 1) // 2
    
    def left_child(self, i: int) -> int:
        """Get left child index"""
        return 2 * i + 1
    
    def right_child(self, i: int) -> int:
        """Get right child index"""
        return 2 * i + 2
    
    def insert(self, key: float, value: int) -> None:
        """Insert a new key-value pair"""
        self.heap.append((key, value))
        self._sift_up(len(self.heap) - 1)
    
    def extract_max(self) -> Tuple[float, int]:
        """Extract the maximum element"""
        if not self.heap:
            return None
        
        max_item = self.heap[0]
        self.heap[0] = self.heap[-1]
        self.heap.pop()
        
        if self.heap:
            self._sift_down(0)
        
        return max_item
    
    def _sift_up(self, i: int) -> None:
        """Move element up to maintain heap property"""
        parent = self.parent(i)
        if i > 0 and self.heap[parent][0] < self.heap[i][0]:
            self.heap[parent], self.heap[i] = self.heap[i], self.heap[parent]
            self._sift_up(parent)
    
    def _sift_down(self, i: int) -> None:
        """Move element down to maintain heap property"""
        max_index = i
        left = self.left_child(i)
        right = self.right_child(i)
        
        if left < len(self.heap) and self.heap[left][0] > self.heap[max_index][0]:
            max_index = left
        
        if right < len(self.heap) and self.heap[right][0] > self.heap[max_index][0]:
            max_index = right
        
        if i != max_index:
            self.heap[i], self.heap[max_index] = self.heap[max_index], self.heap[i]
            self._sift_down(max_index)


# k-d Tree implementation
class KDTreeNode:
    """Node in a k-d tree."""
    
    def __init__(self, point: np.ndarray, value: int, axis: int):
        self.point = point  # Embedding vector
        self.value = value  # Document index
        self.axis = axis    # Split axis
        self.left = None    # Left child
        self.right = None   # Right child


class KDTreeIndex(IndexingStructure):
    """k-d Tree implementation for spatial indexing."""
    
    def __init__(self):
        super().__init__("k-d Tree")
        self.root = None
        self.embeddings = None
        self.documents = None
        self.dim = None
    
    def _build_implementation(self, embeddings: np.ndarray, documents: List[Dict[str, Any]]) -> None:
        """Build k-d tree from embeddings."""
        self.embeddings = embeddings
        self.documents = documents
        self.dim = embeddings.shape[1]
        
        # Build the tree
        points = [(embeddings[i], i) for i in range(len(embeddings))]
        self.root = self._build_tree(points, 0)
    
    def _build_tree(self, points: List[Tuple[np.ndarray, int]], depth: int) -> Optional[KDTreeNode]:
        """Recursively build the k-d tree."""
        if not points:
            return None
        
        # Select axis based on depth
        axis = depth % self.dim
        
        # Sort points and find median
        points.sort(key=lambda x: x[0][axis])
        median_idx = len(points) // 2
        
        # Create node and construct subtrees
        node = KDTreeNode(points[median_idx][0], points[median_idx][1], axis)
        node.left = self._build_tree(points[:median_idx], depth + 1)
        node.right = self._build_tree(points[median_idx + 1:], depth + 1)
        
        return node
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[float, int]]:
        """Search for k nearest neighbors to query_embedding."""
        if not self.root:
            return []
        
        # Use a max heap to find k nearest neighbors
        heap = BinaryMaxHeap()
        
        def search_recursive(node, depth=0):
            if not node:
                return
            
            # Compute distance from query to current node
            dist = np.linalg.norm(query_embedding - node.point)
            
            # If we have less than k points, add this point to the heap
            if len(heap.heap) < k:
                heap.insert(dist, node.value)
            # Otherwise, replace the maximum distance point if this one is closer
            elif dist < heap.heap[0][0]:
                heap.extract_max()
                heap.insert(dist, node.value)
            
            # Select axis based on depth
            axis = depth % self.dim
            
            # Recursively search the half of the tree that contains the target
            if query_embedding[axis] < node.point[axis]:
                search_recursive(node.left, depth + 1)
                # If the distance to the splitting plane is less than the current maximum distance,
                # we also need to check the other side of the plane
                if len(heap.heap) < k or abs(query_embedding[axis] - node.point[axis]) < heap.heap[0][0]:
                    search_recursive(node.right, depth + 1)
            else:
                search_recursive(node.right, depth + 1)
                if len(heap.heap) < k or abs(query_embedding[axis] - node.point[axis]) < heap.heap[0][0]:
                    search_recursive(node.left, depth + 1)
        
        # Start the recursive search
        search_recursive(self.root)
        
        # Extract results from the heap
        results = []
        while heap.heap:
            dist, idx = heap.extract_max()
            sim_score = 1.0 / (1.0 + dist)  # Convert distance to similarity score
            results.append((sim_score, idx))
        
        # Reverse to get highest similarity first
        return sorted(results, key=lambda x: x[0], reverse=True)


# Range Tree implementation for multidimensional data
class RangeTreeNode:
    """Node in a Range Tree."""
    
    def __init__(self, points: List[Tuple[np.ndarray, int]], dimension: int, max_dim: int):
        self.dimension = dimension
        self.max_dim = max_dim
        self.left = None
        self.right = None
        self.next_dimension = None
        
        if not points:
            return
        
        # Sort points by current dimension
        points.sort(key=lambda x: x[0][dimension])
        
        # Find median point
        median_idx = len(points) // 2
        self.median_value = points[median_idx][0][dimension]
        self.median_point = points[median_idx][0]
        self.value = points[median_idx][1]
        
        # Build left and right subtrees
        if median_idx > 0:
            self.left = RangeTreeNode(points[:median_idx], dimension, max_dim)
        if median_idx + 1 < len(points):
            self.right = RangeTreeNode(points[median_idx + 1:], dimension, max_dim)
        
        # Build next dimension tree if needed
        if dimension + 1 < max_dim:
            self.next_dimension = RangeTreeNode(points, dimension + 1, max_dim)


class RangeTreeIndex(IndexingStructure):
    """Range Tree implementation for multidimensional indexing."""
    
    def __init__(self, dimensions_to_index: int = 3):
        super().__init__("Range Tree")
        self.root = None
        self.embeddings = None
        self.documents = None
        self.dimensions_to_index = dimensions_to_index  # Number of dimensions to index
    
    def _build_implementation(self, embeddings: np.ndarray, documents: List[Dict[str, Any]]) -> None:
        """Build Range Tree from embeddings."""
        self.embeddings = embeddings
        self.documents = documents
        
        # Reduce dimensionality for indexing if needed
        if embeddings.shape[1] > self.dimensions_to_index:
            # Use PCA to reduce dimensions
            from sklearn.decomposition import PCA
            pca = PCA(n_components=self.dimensions_to_index)
            reduced_embeddings = pca.fit_transform(embeddings)
        else:
            reduced_embeddings = embeddings
            self.dimensions_to_index = embeddings.shape[1]
        
        # Build the tree
        points = [(reduced_embeddings[i], i) for i in range(len(reduced_embeddings))]
        self.root = RangeTreeNode(points, 0, self.dimensions_to_index)
    
    def _search_nearest(self, node: RangeTreeNode, query: np.ndarray, k: int, candidates: List[Tuple[float, int]]) -> None:
        """Recursively search for nearest neighbors in the Range Tree."""
        if not node:
            return
        
        # Compute distance to the median point
        if hasattr(node, 'median_point'):
            dist = np.linalg.norm(query[:self.dimensions_to_index] - node.median_point)
            
            # Add to candidates if close enough
            if len(candidates) < k:
                heapq.heappush(candidates, (-dist, node.value))
            elif dist < -candidates[0][0]:
                heapq.heappushpop(candidates, (-dist, node.value))
        
        # Determine which subtree to search first
        if not hasattr(node, 'median_value') or query[node.dimension] < node.median_value:
            self._search_nearest(node.left, query, k, candidates)
            
            # If we haven't found k points or the distance to the splitting plane
            # is less than the distance to the farthest point in our candidates,
            # we need to search the other subtree as well
            if len(candidates) < k or abs(query[node.dimension] - node.median_value) < -candidates[0][0]:
                self._search_nearest(node.right, query, k, candidates)
        else:
            self._search_nearest(node.right, query, k, candidates)
            if len(candidates) < k or abs(query[node.dimension] - node.median_value) < -candidates[0][0]:
                self._search_nearest(node.left, query, k, candidates)
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[float, int]]:
        """Search for k nearest neighbors to query_embedding."""
        if not self.root:
            return []
        
        # Reduce query dimensionality if needed
        if query_embedding.shape[0] > self.dimensions_to_index:
            query = query_embedding[:self.dimensions_to_index]
        else:
            query = query_embedding
        
        # Search for candidates using the tree
        candidates = []
        self._search_nearest(self.root, query, k * 2, candidates)  # Get more candidates for reranking
        
        # Rerank candidates using the full embeddings
        results = []
        for _, idx in candidates:
            sim_score = np.dot(query_embedding, self.embeddings[idx]) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(self.embeddings[idx])
            )
            results.append((float(sim_score), idx))
        
        # Return top k
        return sorted(results, key=lambda x: x[0], reverse=True)[:k]


# Factory to get different index structures
def get_index_structure(name: str) -> IndexingStructure:
    """Get an index structure by name."""
    structures = {
        "redblack": RedBlackTreeIndex,
        "skiplist": SkipListIndex,
        "kdtree": KDTreeIndex,
        "rangetree": RangeTreeIndex,
    }
    
    if name not in structures:
        raise ValueError(f"Unknown index structure: {name}. Available: {list(structures.keys())}")
    
    return structures[name]()


# Benchmark different index structures
def benchmark_structures(embeddings: np.ndarray, documents: List[Dict[str, Any]], query_embeddings: np.ndarray, k: int = 10) -> Dict[str, Dict[str, float]]:
    """Benchmark different index structures."""
    structures = {
        "redblack": RedBlackTreeIndex(),
        "skiplist": SkipListIndex(),
        "kdtree": KDTreeIndex(),
        "rangetree": RangeTreeIndex(),
    }
    
    results = {}
    
    for name, structure in structures.items():
        print(f"Building {name} index...")
        structure.build(embeddings, documents)
        
        print(f"Searching with {name} index...")
        query_times = []
        
        for query in query_embeddings:
            start_time = time.time()
            structure.search(query, k=k)
            query_time = time.time() - start_time
            query_times.append(query_time)
        
        results[name] = {
            "build_time": structure.build_time,
            "avg_query_time": sum(query_times) / len(query_times),
            "min_query_time": min(query_times),
            "max_query_time": max(query_times),
        }
        
        print(f"  Build time: {structure.build_time:.4f}s")
        print(f"  Avg query time: {results[name]['avg_query_time']:.4f}s")
    
    return results
