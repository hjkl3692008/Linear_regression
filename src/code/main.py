import linear_tools as lt
import file_tools as ft


def plot_house_heat_map():
    house_data = ft.load_house(trans=False)
    lt.heat_map(house_data, title='correlation matrix of house features')


if __name__ == '__main__':
    plot_house_heat_map()
