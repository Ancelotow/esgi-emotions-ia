from matplotlib import pyplot as plt


def show_digit_vector(vector):
    fig, ax = plt.subplot()
    ax.matshow(vector.reshape(28, 28))
    fig.tight_layout()
    plt.show()


def one_hot_encoder(value, values):
    return [1 if x == value else 0 for x in values]