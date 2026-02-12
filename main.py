"""
Algorithms II — BSD 404
Dynamic programming, advanced graph algorithms, NP-completeness, amortized analysis.
"""

import heapq
from functools import lru_cache

# ── Dynamic Programming ───────────────────────────────────────────────────

def knapsack_01(weights, values, capacity):
    """0/1 knapsack — O(n*W) DP."""
    n  = len(weights)
    dp = [[0]*(capacity+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for w in range(capacity+1):
            dp[i][w] = dp[i-1][w]
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i][w], dp[i-1][w-weights[i-1]] + values[i-1])

    # backtrack
    w, items = capacity, []
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            items.append(i-1); w -= weights[i-1]
    return dp[n][capacity], list(reversed(items))

def lcs(a, b):
    """Longest Common Subsequence."""
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if a[i-1] == b[j-1]: dp[i][j] = dp[i-1][j-1] + 1
            else:                 dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    # reconstruct
    seq, i, j = [], m, n
    while i > 0 and j > 0:
        if a[i-1] == b[j-1]: seq.append(a[i-1]); i -= 1; j -= 1
        elif dp[i-1][j] > dp[i][j-1]: i -= 1
        else:                          j -= 1
    return dp[m][n], ''.join(reversed(seq))

def edit_distance(a, b):
    """Levenshtein distance."""
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[m][n]

def matrix_chain(dims):
    """Matrix chain multiplication — minimum scalar multiplications."""
    n  = len(dims) - 1
    dp = [[0]*n for _ in range(n)]
    for length in range(2, n+1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            for k in range(i, j):
                cost = dp[i][k] + dp[k+1][j] + dims[i]*dims[k+1]*dims[j+1]
                dp[i][j] = min(dp[i][j], cost)
    return dp[0][n-1]

# ── Advanced Graph Algorithms ─────────────────────────────────────────────

def bellman_ford(graph, src, n):
    """Bellman-Ford — handles negative weights, detects negative cycles."""
    dist = [float('inf')] * n
    dist[src] = 0
    for _ in range(n - 1):
        for u, v, w in graph:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
    # check negative cycle
    for u, v, w in graph:
        if dist[u] != float('inf') and dist[u] + w < dist[v]:
            return None, True  # negative cycle
    return dist, False

def floyd_warshall(adj):
    """All-pairs shortest paths."""
    n    = len(adj)
    dist = [row[:] for row in adj]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    return dist

# ── Amortized Analysis Demo ───────────────────────────────────────────────

class DynamicArray:
    """Demonstrates amortized O(1) append via doubling strategy."""
    def __init__(self):
        self._data     = [None]
        self._size     = 0
        self._capacity = 1
        self._resizes  = 0

    def append(self, val):
        if self._size == self._capacity:
            self._capacity *= 2
            new = [None] * self._capacity
            for i in range(self._size): new[i] = self._data[i]
            self._data = new
            self._resizes += 1
        self._data[self._size] = val
        self._size += 1

    def stats(self):
        return {"size": self._size, "capacity": self._capacity, "resizes": self._resizes}

if __name__ == "__main__":
    print("=== Dynamic Programming ===")

    weights = [2, 3, 4, 5]; values = [3, 4, 5, 6]; cap = 8
    val, items = knapsack_01(weights, values, cap)
    print(f"  0/1 Knapsack (cap={cap}): value={val}  items={items}")

    length, subseq = lcs("ABCBDAB", "BDCAB")
    print(f"  LCS('ABCBDAB','BDCAB'): length={length}  '{subseq}'")

    print(f"  Edit distance('kitten','sitting') = {edit_distance('kitten','sitting')}")

    dims = [30, 35, 15, 5, 10, 20, 25]
    print(f"  Matrix chain {dims}: min multiplications = {matrix_chain(dims)}")

    print("\n=== Bellman-Ford ===")
    edges = [(0,1,4),(0,2,5),(1,2,-3),(2,3,4),(3,1,2)]
    dist, neg_cycle = bellman_ford(edges, 0, 4)
    if neg_cycle: print("  Negative cycle detected!")
    else:         print(f"  Shortest from 0: {dist}")

    print("\n=== Floyd-Warshall (all-pairs) ===")
    INF = float('inf')
    adj = [[0,3,INF,7],[8,0,2,INF],[5,INF,0,1],[2,INF,INF,0]]
    fw  = floyd_warshall(adj)
    for i, row in enumerate(fw):
        print(f"  From {i}: {[round(x,1) if x != INF else '∞' for x in row]}")

    print("\n=== Dynamic Array (Amortized Analysis) ===")
    da = DynamicArray()
    for i in range(1000): da.append(i)
    s = da.stats()
    print(f"  After 1000 appends: {s}")
    print(f"  Resizes: {s['resizes']} (log₂(1000) ≈ 10)")
