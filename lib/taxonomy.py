import networkx as nx

class Taxonomy:
    """A class for operations on taxonomy"""

    def __init__(self, graph):
        self.graph = graph

    @classmethod
    def from_data(cls, data):
        """Create data for the data present in full.json file
        """
        graph = nx.DiGraph()
        nodes = []
        edges = []
        for elem in data:
            nodes.append(elem)
            # we prefer to have child relation, so we put parent first
            edges.extend((parent, elem) for parent in data[elem]['parents'])
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        return cls(graph)

    def ancestors(self, cat):
        return nx.ancestors(self.graph, cat)

    def descendants(self, cat):
        return nx.descendants(self.graph, cat)
    
    def cats_complete(self, cats):
        """Add ancestors categories to a list of cats
        """
        scats = set(cats)
        for cat in cats:
            scats.update(self.ancestors(cat))
        return scats

    def cats_compat(self, cats):
        """Define the compatible and incompatible categories for a list of categories

        The idea is that we don't want to penalize categories that are found by the model,
        because it goes deeper than what the dataset is providing
        (database is not necessarily complete)
        """
        # we have two types of categories, the one that have no children in the set
        # they are leaf
        scats = frozenset(cats)
        leafs = frozenset(c for c in cats if not (frozenset(self.descendants(c)) & scats))
        # and the one that have children in the set
        parents = frozenset(c for c in cats if c not in leafs)
        # A category not in the set is compatible
        # if it is descendant of one of the leaf
        compatibles = frozenset(descendant for c in leafs for descendant in self.descendants(c))
        # compatibles = frozenset(c for c in self.graph if frozenset(self.ancestors(c)) & leafs)
        return compatibles

    def cats_filter(self, categories_vocab):
        """intersect categories vocabulary with taxonomy known categories"""
        return [c for c in categories_vocab if c in self.graph]