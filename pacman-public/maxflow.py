"""
maxflow.py sourced from https://github.com/abhinavcreed13/ai-capture-the-flag-pacman-contest/blob/main/myTeam.py#L116
from lines 116 to 231.
Represents helper functions for the Ford-Fulkerson algorithm and flow graph representations.

NOTE
UNUSED in the final submission (reverted to food densities due to crashing).
"""
from util import Queue

class Edge(object):
  """
  Edge object for a flow graph.
  Code sourced:
  https://github.com/abhinavcreed13/ai-capture-the-flag-pacman-contest/blob/main/myTeam.py#L116
  """
  def __init__(self, u, v, w):
    self.source = u
    self.target = v
    self.capacity = w

  def __repr__(self):
    return f"{self.source}->{self.target}:{self.capacity}"

  def __eq__(self, other):
    return self.source == other.source and self.target == other.target

  def __hash__(self):
    return hash(f"{self.source}->{self.target}:{self.capacity}")

class FlowNetwork(object):
  """
  Represent a flow network
  Code sourced:
  https://github.com/abhinavcreed13/ai-capture-the-flag-pacman-contest/blob/main/myTeam.py#L131

  """
  def __init__(self):
    self.adj = {}
    self.flow = {}

  def AddVertex(self, vertex):
    """
    Setter method for the adjacency matrix
    """
    self.adj[vertex] = []

  def GetEdges(self, v):
    """
    Get the edges from the adjacency matrix connecting to v
    """
    return self.adj[v]

  def AddEdge(self, u, v, w=0):
    """
    Add an edge going from vertex u to v with capacity 0 by default.
    """
    # can't have a self edge
    if u == v:
      raise ValueError("u == v")
    
    # create the edge between vertices u and v
    edge = Edge(u, v, w)

    # create a residual edge going backwards
    redge = Edge(v, u, w)

    # add the corresponding residual edge
    edge.redge = redge

    # add the original edge to the residual edge
    redge.redge = edge
    self.adj[u].append(edge)
    self.adj[v].append(redge)

    # Intialize all flows to zero
    self.flow[edge] = 0
    self.flow[redge] = 0

  def FindPath(self, start, goal, path=[]):
    """
    Run a BFS as a inside algorithm to find the path in a Max Flow graph
    """
    node, pathCost = start, 0
    frontier = Queue()
    visited = set()

    if start == goal:
      return path

    while node != goal:
      successors = [(edge.target, edge) for edge in self.GetEdges(node)]
      for successor, edge in successors:
        residual = edge.capacity - self.flow[edge]
        intPath = (edge, residual)
        if residual > 0 and not intPath in path and intPath not in visited:
          visited.add(intPath)
          frontier.push((successor, path + [(edge, residual)], pathCost + 1))

      if frontier.isEmpty():
        return None
      else:
        node, path, pathCost = frontier.pop()

    return path

  def MaxFlow(self, source, target):
    """
    Find the MaxFlow + a variable to keep edges which are reachable from sink point T.
    Code sourced:
    https://github.com/abhinavcreed13/ai-capture-the-flag-pacman-contest/blob/main/myTeam.py#L182
    """
    targetEdges = {}
    path = self.FindPath(source, target)
    while path is not None:
      targetEdges[path[0]] = path
      flow = min(res for edge, res in path)
      for edge, _ in path:
        self.flow[edge] += flow
        self.flow[edge.redge] -= flow

      path = self.FindPath(source, target)
    maxflow = sum([self.flow[edge] for edge in self.GetEdges(source)])
    return maxflow, targetEdges

  def FindBottlenecks(self, source, target):
    """
    Find Bottleneck position using the Ford Fulkerson Algorithm. The idea is, We have the edges
    which are reachable from  sink point T. Get the edges which are reachable from source S.
    The bottleneck positions are nodes connecting these both sets.

    Code sourced:
    https://github.com/abhinavcreed13/ai-capture-the-flag-pacman-contest/blob/main/myTeam.py#L199
    """
    _, targetEdges = self.MaxFlow(source, target)
    paths = targetEdges.values()

    bottlenecks = []
    for path in paths:
      for edge, _ in path:
        if self.FindPath(source, edge.target) is None:
          bottlenecks.append(edge.source)
          break
    return bottlenecks

class SortedEdges(dict):
  '''
  Extends dictionary to contain methods to sort edges for the Max Flow Problem
  Code sourced:
  https://github.com/abhinavcreed13/ai-capture-the-flag-pacman-contest/blob/main/myTeam.py#L216

  '''

  def __init__(self, *args, **kwargs):
    dict.__init__(self, *args, **kwargs)

  def __getitem__(self, key):
    return dict.__getitem__(self, tuple(sorted(key)))

  def __setitem__(self, key, val):
    return dict.__setitem__(self, tuple(sorted(key)), val)

  def __contains__(self, key):
    return dict.__contains__(self, tuple(sorted(key)))
