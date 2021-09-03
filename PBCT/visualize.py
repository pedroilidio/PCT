import graphviz
from matplotlib import colors
import colorsys
from matplotlib.cm import get_cmap


def lightness(c):
    """Calculate lighnex of hex-formatted color."""
    return colorsys.rgb_to_hls(*colors.hex2color(c))[1]


# FIXME: Still in node classes format.
def add_node(node, graph, node_cmap, leaf_cmap, parent_id=None):
    node_id = str(hash(node))

    if node['is_leaf']:
        fillcolor = leaf_cmap(node.Ymean)
        fontcolor = ('white', 'black')[lightness(fillcolor) > .5]
        graph.attr('node', shape='ellipse',
                   fillcolor=fillcolor, fontcolor=fontcolor)
    else:
        fillcolor = node_cmap(node.split_quality)
        fontcolor = ('white', 'black')[lightness(fillcolor) > .5]
        graph.attr('node', shape='box',
                   fillcolor=fillcolor, fontcolor=fontcolor)

    graph.node(node_id, str(node))

    if parent_id is not None:
        graph.edge(parent_id, node_id)

    if not node.is_leaf:
        for child in (node.little_child, node.big_child):
            add_node(child, graph, parent_id=node_id,
                     node_cmap=node_cmap, leaf_cmap=leaf_cmap)


def render_tree(node, leaf_cmap='RdYlGn', node_cmap='YlOrBr',
                format='png', *args, **kwargs):
    if isinstance(leaf_cmap, str):
        leaf_cmap=get_cmap(leaf_cmap)
    if isinstance(node_cmap, str):
        node_cmap=get_cmap(node_cmap)
    leaf_cmap_hex = lambda x: colors.rgb2hex(leaf_cmap(x))
    node_cmap_hex = lambda x: colors.rgb2hex(node_cmap(x))

    graph = graphviz.Digraph(*args, format=format, **kwargs)
    graph.attr('node', style='filled')

    add_node(node, graph,
             leaf_cmap=leaf_cmap_hex,
             node_cmap=node_cmap_hex)

    # Use name attribute to set output location.
    outpath = graph.render()
    return outpath
