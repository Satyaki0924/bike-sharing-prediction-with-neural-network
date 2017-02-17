"""
This is a project by Satyaki Sanyal.
This project must be used for educational purposes only.

Follow me on:
LinkedIn - https://www.linkedin.com/in/satyaki-sanyal-708424b7/
Github - https://github.com/Satyaki0924/
Researchgate - https://www.researchgate.net/profile/Satyaki_Sanyal
"""
from functions.Predict import Predict


def run():
    set_epochs = 5000
    set_hidden_layers = 222
    set_learning_rate = 0.001
    Predict(set_epochs, set_learning_rate, set_hidden_layers)._predict()


if __name__ == '__main__':
    run()
