# plot rope diagram for tuple in first 8 positions
#
import matplotlib.pyplot as plt
import numpy as np
import argparse, sys

tupel = [1, 1]

parser = argparse.ArgumentParser()
parser.add_argument("--norope", action="store_true", help="use orig transformer emded")
parser.add_argument("args", nargs="*", help="optional x, y float tuple")
args = parser.parse_args()

if len(args.args) > 0:
   tupel[0] = float(args.args[0])
if len(args.args) > 1:
   tupel[1] = float(args.args[1])

np.set_printoptions(suppress=True, precision=4, edgeitems=4)

def positional_encoding(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denom = np.power(n, 2*i/d)
            val = k/denom
            P[k, 2*i] = np.cos(val)
            P[k, 2*i+1] = np.sin(val)
    return P
 
seq_len=8 
dim=4096 
P = positional_encoding(seq_len=seq_len, d=dim, n=10000)
print('positional:', P)

# create row of alternating values
q_row = tupel * int(dim/2)
Q = np.array([]).reshape(0, dim)
for k in range(seq_len):
    Q = np.append(Q, [q_row], axis=0) # append row to matrix

# add positional values to query matrix
for k in range(seq_len):
    for i in np.arange(int(dim/2)):
        fcr = P[k, 2*i]
        fci = P[k, 2*i+1]
        if args.norope:
            Q[k, 2*i] += fcr
            Q[k, 2*i+1] += fci
        else:
            v0 = Q[k, 2*i]
            v1 = Q[k, 2*i+1]
            Q[k, 2*i] = v0 * fcr - v1 * fci
            Q[k, 2*i+1] = v0 * fci + v1 * fcr

print('rotated_query:', Q)

# circle center
center = (0, 0)

# get x/y arrays for first 8 points to plot on circle
cnt=8
ppx = [[0 for _ in range(seq_len)] for _ in range(cnt)]
ppy = [[0 for _ in range(seq_len)] for _ in range(cnt)]

for k in range(seq_len):
    for i in np.arange(cnt):
        ppx[i][k] = center[0] + (Q[k, 2*i]/np.sqrt(2))
        ppy[i][k] = center[1] + (Q[k, 2*i+1]/np.sqrt(2))

# take radius from first point
radius = np.sqrt(ppx[0][0]*ppx[0][0] + ppy[0][0]*ppy[0][0])

print('x-cos:', np.array(ppx))
print('y-sin:', np.array(ppy))

# labels for points
labels = ["Pos0", "Pos1", "Pos2", "Pos3", "Pos4", "Pos5", "Pos6", "Pos7"]

# color gradient
start_color = [1, 0, 0] # red
end_color = [1, 1, 0]   # yellow

# generate intermediate colors
colors = [tuple(np.linspace(start_color, end_color, num=cnt)[i]) for i in range(cnt)]

# create the plot
fig, ax = plt.subplots(figsize=(6, 6))

for i in reversed(range(cnt)):
    ax.plot(ppx[i], ppy[i], 'o', color=colors[i])

# plot circle outline in blue
circle = plt.Circle(center, radius, color='b', fill=False)
ax.add_patch(circle)

# equal aspect ratio to make circle round
ax.set_aspect('equal', adjustable='box')

# add point labels outside of circle
for i, txt in enumerate(labels):
    xt = 10 if ppx[0][i] >= 0 else -20
    yt = 10 if ppy[0][i] >= 0 else -15
    ax.annotate(txt, (ppx[0][i], ppy[0][i]), textcoords="offset points", xytext=(xt, yt), ha='center')

# axis limits
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)

# x and y axes
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)

# title and labels
ax.set_title(f"Positional Encoding for Tupel (x={tupel[0]}, y={tupel[1]})")
ax.set_xlabel("X (cos)")
ax.set_ylabel("Y (sin)")

plt.show()
