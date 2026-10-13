"""DAGs for Report 5.

Nodes: Z random assignment, D climate-policies video (vs. control), Y supports the carbon tax
with cash transfers, S interest in politics (baseline), X other baseline covariates, M
believes own household would lose (post-treatment).

DAG 1  the experiment: D's only parent is Z, so there is no backdoor path.
DAG 2  a biased sample: S -> D opens the backdoor D <- S -> Y (red); {S} blocks it.
DAG 3  a mediator: D -> M -> Y (green); M is post-treatment and, with X -> M, a collider.

Outputs: fig_dag1_experiment.pdf, fig_dag2_biased.pdf, fig_dag3_mediator.pdf, fig_sutva.pdf
"""
import graphviz
import networkx as nx
import pandas as pd

from Scripts.utils import fig_path, show

TITLE = "DAGs 1-3; NetworkX checks"

COLORS = ['#00B0F0', '#FF0000', '#00A651']        # blue (nodes), red (backdoor), green (mediated)

NODES = {
    'Z': ('Z: random assignment\n(coin flip)', '0,1!'),
    'D': ('D: climate-policies video\n(vs. control)', '2.6,1!'),
    'Y': ('Y: supports carbon tax\nwith cash transfers', '6.4,1!'),
    'S': ('S: interest in politics\n(baseline)', '4.5,2.6!'),
    'X': ('X: other baseline covariates\n(country, transport access,\nhousehold size, leaning, ...)', '4.5,-0.7!'),
}
NODES3 = dict(NODES)
NODES3.update({
    'Z': ('Z: random assignment\n(coin flip)', '0,1.9!'),
    'D': ('D: climate-policies video\n(vs. control)', '2.9,1.9!'),
    'M': ('M: believes own household\nwould lose (post-treatment)', '4.9,0.85!'),
    'Y': ('Y: supports carbon tax\nwith cash transfers', '7.4,1.9!'),
    'S': ('S: interest in politics\n(baseline)', '5.2,3.3!'),
    'X': ('X: other baseline covariates\n(country, transport access,\nhousehold size, leaning, ...)', '4.9,-0.6!'),
})
DAG1_EDGES = ['ZD', 'DY', 'SY', 'XY']
DAG2_EDGES = DAG1_EDGES + ['SD']
DAG3_EDGES = ['ZD', 'DY', 'DM', 'MY', 'SY', 'XY', 'XM']


def draw_dag(nodes, edges, fname, colored=None, dashed=(), title=None):
    """nodes {name: (label, 'x,y!')} for the neato engine; edges like 'ZD' = Z -> D."""
    colored = colored or {}
    g = graphviz.Digraph(format='pdf', engine='neato')
    g.attr(fontname='Helvetica', fontsize='14', labelloc='t', label=title or '')
    g.attr('node', shape='box', style='rounded,filled', fontname='Helvetica', fontsize='12',
           fillcolor='#E8F7FD', color=COLORS[0])
    g.attr('edge', penwidth='1.6', arrowsize='0.9')
    for n, (label, pos) in nodes.items():
        g.node(n, label=label, pos=pos)
    for e in edges:
        g.edge(e[0], e[1], color=colored.get(e, 'black'),
               penwidth='3' if e in colored else '1.6',
               style='dashed' if e in dashed else 'solid')
    g.render(str(fig_path(fname)), cleanup=True)          # writes fname.pdf
    print(f"wrote {fname}.pdf")
    return g


def sutva_diagram():
    g = graphviz.Digraph(format='pdf', engine='neato')
    g.attr(fontname='Helvetica', fontsize='13', labelloc='t',
           label="SUTVA: each respondent answers alone online; no arrow from one respondent's "
                 "video to another's answers")
    g.attr('node', shape='box', style='rounded,filled', fontname='Helvetica', fontsize='12',
           fillcolor='#E8F7FD', color='#00B0F0')
    g.attr('edge', penwidth='1.6')
    for who, y in [('i', '2'), ('j', '0')]:
        g.node(f'Z{who}', f'Z{who}: random assignment\n(respondent {who})', pos=f'0,{y}!')
        g.node(f'D{who}', f'D{who}: video shown\nto respondent {who}', pos=f'2.8,{y}!')
        g.node(f'Y{who}', f'Y{who}: support of\nrespondent {who}', pos=f'5.6,{y}!')
        g.edge(f'Z{who}', f'D{who}')
        g.edge(f'D{who}', f'Y{who}')
    g.edge('Di', 'Yj', style='dashed', color='#FF0000', label='  ruled out',
           fontcolor='#FF0000', fontsize='11')
    g.edge('Dj', 'Yi', style='dashed', color='#FF0000')
    g.render(str(fig_path('fig_sutva')), cleanup=True)
    print("wrote fig_sutva.pdf")
    return g


def d_separated(G, x, y, z):
    f = getattr(nx, "is_d_separator", None) or nx.d_separated   # renamed in NetworkX 3.3
    return f(G, {x}, {y}, set(z))


def describe(name, edges, controls_to_try):
    G = nx.DiGraph([(e[0], e[1]) for e in edges])
    print(f"===== {name}")
    print("  acyclic:", nx.is_directed_acyclic_graph(G), "  parents of D:", sorted(G.predecessors('D')))
    for p in nx.all_simple_paths(G.to_undirected(), 'D', 'Y'):
        arrows = "".join(f" {'->' if G.has_edge(a, b) else '<-'} {b}" for a, b in zip(p, p[1:]))
        colliders = [b for a, b, c in zip(p, p[1:], p[2:]) if G.has_edge(a, b) and G.has_edge(c, b)]
        if all(G.has_edge(a, b) for a, b in zip(p, p[1:])):
            kind = "causal (directed D -> ... -> Y)"
        elif G.has_edge(p[1], 'D'):
            kind = "backdoor" + (f", blocked by collider {colliders}" if colliders else ", open")
        else:
            kind = f"non-causal, blocked by collider {colliders}"
        print(f"  path: {p[0]}{arrows:<22}  [{kind}]")
    G_bd = G.copy()
    G_bd.remove_edges_from(list(G.out_edges('D')))        # backdoor criterion
    for z in controls_to_try:
        ok = d_separated(G_bd, 'D', 'Y', z)
        print(f"  control for {set(z) or '{}'}: backdoor paths {'all blocked' if ok else 'OPEN'}")
    return G


def run(ctx):
    graphs = [
        draw_dag(NODES, DAG1_EDGES, 'fig_dag1_experiment',
                 title='DAG 1 - the experiment: D has one parent, the random assignment'),
        draw_dag(NODES, DAG2_EDGES, 'fig_dag2_biased', colored={'SD': COLORS[1], 'SY': COLORS[1]},
                 title='DAG 2 - the biased sample: S drives take-up; backdoor path D <- S -> Y in red'),
        draw_dag(NODES3, DAG3_EDGES, 'fig_dag3_mediator',
                 colored={'DM': COLORS[2], 'MY': COLORS[2]}, dashed=('XM',),
                 title='DAG 3 - a mediator: D -> M -> Y (green); M is post-treatment, do not control for it'),
        # sutva_diagram()
        ]
    for g in graphs:
        show(g, terminal=False)            # drawn in the notebook; the PDFs are saved either way

    describe("DAG 1 - experiment", DAG1_EDGES, [[]])
    G2 = describe("DAG 2 - biased sample", DAG2_EDGES, [[], ['S'], ['X']])
    describe("DAG 3 - mediator", DAG3_EDGES, [[]])
    # Conditioning on M in DAG 3 (without D -> Y): blocks the chain D -> M -> Y but opens the
    # collider path D -> M <- X -> Y; controlling for X as well closes it again.
    G3_noDY = nx.DiGraph([(e[0], e[1]) for e in DAG3_EDGES if e != 'DY'])
    for z in [[], ['M'], ['M', 'X']]:
        print(f"DAG 3 without D -> Y: D and Y d-separated given {set(z) or '{}'}? "
              f"{d_separated(G3_noDY, 'D', 'Y', z)}")
    print("\nAdjacency matrix of DAG 2 (rows = source, columns = destination):")
    order = ['Z', 'S', 'X', 'D', 'Y']
    show(pd.DataFrame(nx.to_numpy_array(G2, nodelist=order, dtype=int), index=order, columns=order))
    return {}


if __name__ == "__main__":
    from Scripts.runner import run_standalone
    run_standalone("dags")
