class MaxHeap:
    def __init__(self):
        self.heap = []

    def parent(self, i):
        return (i - 1) // 2

    def left_child(self, i):
        return 2 * i + 1

    def right_child(self, i):
        return 2 * i + 2

    def swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]

    def insert(self, key):
        """
        Insert a new element into the max heap
        """
        self.heap.append(key)
        self._heapify_up(len(self.heap) - 1)

    def _heapify_up(self, i):
        """
        Maintain heap property after insertion
        """
        while i > 0 and self.heap[self.parent(i)] < self.heap[i]:
            parent_idx = self.parent(i)
            self.swap(i, parent_idx)
            i = parent_idx

    def extract_max(self):
        """
        Remove and return the maximum element
        """
        if not self.heap:
            return None
        
        if len(self.heap) == 1:
            return self.heap.pop()
        
        max_val = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._heapify_down(0)
        
        return max_val

    def _heapify_down(self, i):
        """
        Maintain heap property after extraction
        """
        max_idx = i
        left = self.left_child(i)
        right = self.right_child(i)
        
        # Find the largest among root, left child, and right child
        if left < len(self.heap) and self.heap[left] > self.heap[max_idx]:
            max_idx = left
        
        if right < len(self.heap) and self.heap[right] > self.heap[max_idx]:
            max_idx = right
        
        # If largest is not the root, swap and continue heapifying
        if max_idx != i:
            self.swap(i, max_idx)
            self._heapify_down(max_idx)

    def get_top_k(self, k):
        """
        Return top k elements while maintaining heap structure
        """
        top_k = []
        heap_copy = self.heap.copy()
        
        for _ in range(min(k, len(self.heap))):
            top_k.append(self.extract_max())
        
        # Restore the heap
        self.heap = heap_copy
        
        return top_k

    def __len__(self):
        return len(self.heap)