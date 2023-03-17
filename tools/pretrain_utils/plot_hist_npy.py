import numpy as np
import matplotlib
import matplotlib.pyplot as plt

font = {'family' : 'normal',
        'size'   : 16}
matplotlib.rc('font', **font)
plt.rcParams["font.family"] = ["Times New Roman"]


facecolor = {"r": tuple(np.asarray((190,0,0))/255),
             "s": tuple(np.asarray((49,130,189))/255),
             "sd": tuple(np.asarray((0,255,0))/255)}
alpha = {"r": 0.5,
         "s": 0.5,
         "sd": 0.5}
bins = np.arange(0, 150, 1)

fig, ax1 = plt.subplots(figsize=(8, 6))

sum_counts = {}
for mode in ["r", "s"]:
    sum_counts[f"{mode}"] = np.load(f"hist_{mode}.npy")
    ax1.hist(bins[:-1], bins, weights=sum_counts[f"{mode}"], color=facecolor[f"{mode}"], log=True, alpha=alpha[f"{mode}"])
ax1.set_ylim([1e-4, 1e-1])
ax1.set_xlim([0, 150])

if False:
    # probability difference (sim-real), but divided by sim to keep it relative: (sim-real)/sim
    prob_diff = (sum_counts["s"]-sum_counts["r"])/sum_counts["s"]
    np.save("prob_diff_s_sd.npy", prob_diff)

    ax2 = ax1.twinx()
    ax2.hist(bins[:-1], bins, weights=prob_diff, color="k", alpha=0.2)
    ax2.set_ylim([0, 1.0])

plt.xlabel("Point range in meter")
plt.ylabel("Point probability")
plt.legend(["Real dataset", "Sim dataset"])
ax1.set_axisbelow(True)
ax1.yaxis.grid(color='gray', linestyle='dashed')
plt.show()