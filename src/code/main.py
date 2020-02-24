import svm_tools as lt
import file_tools as ft


def plot_cancer_heat_map():
    cancer_data = ft.load_cancer(trans=False)
    lt.heat_map(cancer_data, title='correlation matrix of cancer features')


if __name__ == '__main__':
    # plot correlation
    plot_cancer_heat_map()