import numpy as np
import matplotlib.pyplot as plt


def sigmoid_membership(x, threshold, tau):
    """
    mu(x) = 1 / (1 + exp(-(x - threshold) / tau))

    x         : value of the already-fuzzified feature (Fuzzifier output, in [0,1])
    threshold : split point learned by the CART at that node (t.threshold[node])
    tau       : width of the transition zone, calibrated locally
                (e.g. Silverman's rule over the samples reaching that node)
    """
    return 1.0 / (1.0 + np.exp(-(x - threshold) / tau))

threshold = 0.5
tau = 0.05

x = np.linspace(0, 1, 1000)
mu = sigmoid_membership(x, threshold, tau)

fig, ax = plt.subplots(figsize=(7, 5))

# curva principal em preto forte
ax.plot(x, mu, color="black", linewidth=2.4, label=r"$\mu(x)$")

# banda de largura tau em torno do threshold (sem valores no gráfico)
ax.axvspan(threshold - tau, threshold + tau, color="gray", alpha=0.18,
           label=r"bandwidth $\tau$")

# threshold (linha vertical, sem anotação de valor)
ax.axvline(threshold, color="black", linestyle="--", linewidth=1.2,
           label="threshold")

ax.set_xlabel(r"$x$ pre-processed")
ax.set_ylabel(r"$\mu(x)$")
ax.set_title("Sigmoid function for fuzzy routing at a node")
ax.set_xlim(0, 1)
ax.set_ylim(-0.02, 1.02)
ax.legend(loc="lower right", fontsize=10, frameon=False)

# sem grade
ax.grid(False)

fig.tight_layout()

# Saída vetorizada em PDF, salva na mesma pasta do script
fig.savefig("sigmoid_membership.pdf")

print("Arquivo gerado: sigmoid_membership.pdf")