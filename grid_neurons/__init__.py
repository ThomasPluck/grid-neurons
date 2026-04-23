"""Grid-neurons: 2D-grid variant of the tree-neurons architecture.

Each grid cell is a first-order Gm-C filter with tanh + bias and has two
inputs (from-left and from-above) and one output (routed to-right and
to-below neighbours). The left shoreline is external input; the bottom
shoreline is the block output. Top and right shorelines are nulled.

See topology.py for connectivity, cell.py for the per-cell primitive,
block.py for the block-level forward/backward with SnAP-1 descendant
traces, and training.py for the BPTT and local-rule training entry
points.
"""
