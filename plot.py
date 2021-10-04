import argparse
import pickle
from time import time

import jax
import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
from flax.core.frozen_dict import FrozenDict

from data import generate_data
from nn import AttentionModel, solve


def test_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--path",
        metavar="P",
        type=str,
        required=True,
        help="Weights/VRP***_train_epoch***.h5, h5 file required",
    )
    parser.add_argument(
        "-b", "--batch", metavar="B", type=int, default=2, help="batch size"
    )
    parser.add_argument(
        "-n",
        "--n_nodes",
        metavar="N",
        type=int,
        default=20,
        help="number of customer nodes, time sequence",
    )
    parser.add_argument(
        "-s",
        "--seed",
        metavar="S",
        type=int,
        default=123,
        help="random seed number for inference, reproducibility",
    )
    parser.add_argument(
        "-e",
        "--embed_dim",
        metavar="S",
        type=int,
        default=128,
        help="embedding dimension",
    )
    parser.add_argument(
        "-t",
        "--txt",
        metavar="T",
        type=str,
        help="if you wanna test out on text file, example: ../OpenData/A-n53-k7.txt",
    )
    parser.add_argument(
        "-d",
        "--decode_type",
        metavar="D",
        type=str,
        required=True,
        choices=["greedy", "sampling"],
        help="greedy or sampling required",
    )
    args = parser.parse_args()

    return args


def get_clean_path(arr):
    """Returns extra zeros from path.
    Dynamical model generates duplicated zeros for several graphs when obtaining partial solutions.
    """
    p1, p2 = 0, 1
    output = []
    while p2 < len(arr):
        if arr[p1] != arr[p2]:
            output.append(arr[p1])
            if p2 == len(arr) - 1:
                output.append(arr[p2])
        p1 += 1
        p2 += 1

    if output[0] != 0:
        output.insert(0, 0)  # insert 0 in 0th of the array
    if output[-1] != 0:
        output.append(0)  # insert 0 at the end of the array
    return output


def plot_route(data, pi, costs, title, idx_in_batch=0):
    """Plots journey of agent
    Args:
            data: dataset of graphs
            pi: (batch, decode_step) # tour
            idx_in_batch: index of graph in data to be plotted
    """

    cost = costs[idx_in_batch]
    # Remove extra zeros
    pi_ = get_clean_path(pi[idx_in_batch])

    coords, demands = data

    depot_xy = coords[idx_in_batch, 0]
    customer_xy = coords[idx_in_batch, 1:]
    demands = demands[idx_in_batch, 1:]
    # customer_labels = ['(' + str(i) + ', ' + str(demand) + ')' for i, demand in enumerate(demands.round(2), 1)]
    customer_labels = ["(" + str(demand) + ")" for demand in demands.round(2)]

    xy = np.concatenate([depot_xy.reshape(1, 2), customer_xy], axis=0)

    # Get list with agent loops in path
    list_of_paths, cur_path = [], []
    for idx, node in enumerate(pi_):

        cur_path.append(node)

        if idx != 0 and node == 0:
            if cur_path[0] != 0:
                cur_path.insert(0, 0)
            list_of_paths.append(cur_path)
            cur_path = []

    path_traces = []
    for i, path in enumerate(list_of_paths, 1):
        coords = xy[[int(x) for x in path]]

        # Calculate length of each agent loop
        lengths = np.sqrt(np.sum(np.diff(coords, axis=0) ** 2, axis=1))
        total_length = np.sum(lengths)

        path_traces.append(
            go.Scatter(
                x=coords[:, 0],
                y=coords[:, 1],
                mode="markers+lines",
                name=f"tour{i} Length = {total_length:.3f}",
                opacity=1.0,
            )
        )

    trace_points = go.Scatter(
        x=customer_xy[:, 0],
        y=customer_xy[:, 1],
        mode="markers+text",
        name="Customer (demand)",
        text=customer_labels,
        textposition="top center",
        marker=dict(size=7),
        opacity=1.0,
    )

    trace_depo = go.Scatter(
        x=[depot_xy[0]],
        y=[depot_xy[1]],
        mode="markers+text",
        name="Depot (capacity = 1.0)",
        text=["1.0"],
        textposition="bottom center",
        marker=dict(size=23),
        marker_symbol="triangle-up",
    )

    layout = go.Layout(
        title=dict(
            text=f"<b>VRP{customer_xy.shape[0]} {title}, Total Length = {cost:.3f}</b>",
            x=0.5,
            y=1,
            yanchor="bottom",
            yref="paper",
            pad=dict(b=10),
        ),  # https://community.plotly.com/t/specify-title-position/13439/3
        # xaxis = dict(title = 'X', ticks='outside'),
        # yaxis = dict(title = 'Y', ticks='outside'),#https://kamino.hatenablog.com/entry/plotly_for_report
        xaxis=dict(
            title="X",
            range=[0, 1.1],
            showgrid=False,
            ticks="outside",
            linewidth=1,
            mirror=True,
        ),
        yaxis=dict(
            title="Y",
            range=[0, 1],
            showgrid=False,
            ticks="outside",
            linewidth=1,
            mirror=True,
        ),
        showlegend=True,
        width=750,
        height=700,
        autosize=True,
        template="plotly_white",
        legend=dict(
            x=1.25,
            xanchor="right",
            y=0,
            yanchor="bottom",
            bordercolor="#444",
            borderwidth=0,
        )
        # legend = dict(x = 0, xanchor = 'left', y =0, yanchor = 'bottom', bordercolor = '#444', borderwidth = 0)
    )

    data = [trace_points, trace_depo] + path_traces
    print("path: ", np.array(pi_))
    fig = go.Figure(data=data, layout=layout)
    fig.show()


if __name__ == "__main__":
    args = test_parser()
    t1 = time()

    rng = jax.random.PRNGKey(0)

    dataset = generate_data(n_samples=1, n_nodes=args.n_nodes, seed=args.seed)

    x = (
        jnp.zeros((args.batch, args.n_nodes, 2)),
        jnp.zeros((args.batch, args.n_nodes)),
    )
    e = jnp.zeros((args.batch, args.n_nodes, args.embed_dim))
    d = jnp.zeros((args.batch, args.embed_dim + 1))
    m = jnp.zeros((args.batch, args.n_nodes, 1), dtype=np.bool)

    model = AttentionModel(
        embed_dim=args.embed_dim,
    )

    # Init weights.
    encoder_state, encoder_params = model.init(rng, x, method=model.encode).pop(
        "params"
    )
    decoder_state, decoder_params = model.init(rng, e, d, m, method=model.decode).pop(
        "params"
    )

    states = (encoder_state, decoder_state)
    params = (encoder_params, decoder_params)

    # If a checkpoint is passed, preload weights from it.
    with open(args.path, "rb") as f:
        params = pickle.load(f)

    encoder_params, decoder_params = params

    deterministic = args.decode_type == "greedy"

    for i, data in dataset.repeat().batch(args.batch).enumerate().as_numpy_iterator():

        variables = (
            FrozenDict({"params": encoder_params, **encoder_state}),
            FrozenDict({"params": decoder_params, **decoder_state}),
        )

        costs, log_probs, pi = solve(
            model,
            variables,
            rng,
            data,
            deterministic=deterministic,
            training=True,
        )

        idx_in_batch = np.argmin(costs, axis=0)
        print("costs:", costs)
        print(
            f"decode type:{args.decode_type}\nminimum cost: {costs[idx_in_batch]:.3f} and idx: {idx_in_batch} out of {args.batch} solutions"
        )
        print(f"{np.array(pi)[idx_in_batch]}\ninference time: {time()-t1}s")
        plot_route(data, pi, costs, "Pretrained", idx_in_batch)
        # costs, _, pi = model(data, return_pi = True)
        # plot_route(data, pi, costs, 'Untrained', idx_min)
        if i == 0:
            break
