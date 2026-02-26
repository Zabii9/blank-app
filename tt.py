import plotly.graph_objects as go

# Define nodes
nodes = ["OB-1", "OB-2", "Store-A", "Store-B", "Store-C", "SKU-101", "SKU-102", "SKU-105"]

# Define links (source index, target index, value)
source = [0, 0, 1, 1]      # OB indexes
target = [2, 2, 3, 4]      # Store indexes
value = [120, 80, 50, 75]  # Sales / Units

# Add SKU flows (Store → SKU)
source += [2, 2, 3, 4]
target += [5, 6, 7, 7]
value += [120, 80, 50, 75]

# Create Sankey diagram
fig = go.Figure(data=[go.Sankey(
    node=dict(label=nodes, pad=15, thickness=20),
    link=dict(source=source, target=target, value=value)
)])
fig.show()