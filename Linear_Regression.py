import matplotlib.pyplot as plt
import numpy as np
import random

features = np.array([1, 2, 3, 5, 6, 7])
labels = np.array([155, 197, 244, 356, 407, 448])


def simple_trick(base_price, price_per_room, num_rooms, price):
    small_random1 = random.random()*0.1
    small_random2 = random.random()*0.1
    predicted_price = base_price + price_per_room * num_rooms
    if price > predicted_price and num_rooms > 0:
        price_per_room += small_random1
        base_price += small_random2
    if price > predicted_price and num_rooms < 0:
        price_per_room -= small_random1
        base_price += small_random2
    if price < predicted_price and num_rooms < 0:
        price_per_room -= small_random1
        base_price -= small_random2
    if price < predicted_price and num_rooms > 0:
        price_per_room += small_random1
        base_price -= small_random2
    return base_price, price_per_room


def square_trick(price, num_of_rooms, bias, price_per_room, learning_rate):
    predicted_price = bias + num_of_rooms * price_per_room

    bias += learning_rate * (price - predicted_price)
    price_per_room += learning_rate * (price - predicted_price) * num_of_rooms

    return bias, price_per_room


def absolute_trick(base_price, price_per_room, num_rooms, price, learning_rate):
    predicated_price = base_price + price_per_room * num_rooms
    if price > predicated_price:
        price_per_room += learning_rate*price_per_room
        base_price += learning_rate
    else:
        price_per_room -= learning_rate*price_per_room
        base_price -= learning_rate
    return base_price, price_per_room


def linear_reg(features, labels, learning_rate, epoch=500):
    price_per_room = random.random()
    bias = random.random()
    for p in range(epoch):
        i = random.randint(0, len(features) - 1)
        num_rooms = features[i]
        price = labels[i]
        bias, price_per_room = square_trick(
            price, num_rooms, bias, price_per_room, learning_rate)
    return bias, price_per_room


bias, price_per_room = linear_reg(features, labels, 0.01, 10000)
print("Linear Regression")
print(price_per_room)
print(bias)


def linear_regression_gd(features, labels, learning_rate=0.01, epochs=1000, plot=False):
    bias = np.random.rand()
    price_per_room = np.random.rand()
    m = len(features)
    for epoch in range(epochs):
        y_pred = bias + price_per_room * features
        error = y_pred - labels
        d_bias = (1/m) * np.sum(error)
        d_price_per_room = (1/m) * np.sum(error * features)
        bias -= learning_rate * d_bias
        price_per_room -= learning_rate * d_price_per_room
        if plot and epoch % 100 == 0:
            plt.scatter(features, labels, color='blue', label='Data')
            plt.plot(features, bias + price_per_room * np.array(features),
                     color='red', label='Prediction')
            plt.xlabel('Number of Rooms')
            plt.ylabel('Price')
            plt.legend()
            plt.title(f"Epoch {epoch}")
            plt.grid(True)
            plt.show()
    return bias, price_per_room


bias, price_per_room = linear_regression_gd(
    features, labels, learning_rate=0.001, epochs=10000)
print("Gradient Descent")
print("Price per room:", price_per_room)
print("Bias:", bias)
