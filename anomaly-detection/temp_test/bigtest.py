# %%
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("pdf")
# %%
xaxis =[1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
yaxis =[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# %%
plt.plot(xaxis, yaxis)
plt.xlabel("X")
plt.ylabel("Y")
# %%
plt.savefig("squares.png")

plt.show()
