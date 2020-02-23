import linear_tools as lt
import file_tools as ft


def plot_house_heat_map():
    house_data = ft.load_house(trans=False)
    lt.heat_map(house_data, title='correlation matrix of house features')


if __name__ == '__main__':
    # plot correlation
    # plot_house_heat_map()

    house_data = ft.load_house(trans=False)
    house_data = lt.delete_abnormal(house_data)
    house_data = lt.contain_price_sq(house_data)

    house_data = ft.df2np(house_data)

    w = lt.linear_regression(house_data)
    print(w)

    lt.plot_scatter_points(house_data, w)
