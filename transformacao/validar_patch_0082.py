import numpy as np
import matplotlib.pyplot as plt


arquivo = "patches/patches_2_5d_0082.npz"

dados = np.load(arquivo)

patches = dados["patches"]

print("Shape:", patches.shape)
print("Tipo:", patches.dtype)
print("Nodule IDs:", dados["nodule_id"])

patch = patches[0]

nomes = [
    "Axial",
    "Coronal",
    "Sagital"
]

for canal in range(3):

    plt.figure(figsize=(5, 5))

    plt.imshow(
        patch[canal],
        cmap="gray",
        vmin=0,
        vmax=1
    )

    plt.title(
        f"Canal {canal} - {nomes[canal]}"
    )

    plt.axis("off")

    plt.show()