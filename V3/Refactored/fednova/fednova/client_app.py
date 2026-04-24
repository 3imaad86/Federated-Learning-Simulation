"""Client Flower pour FedNova.

Le client renvoie ses poids locaux et `tau_i` ; la normalisation FedNova
est faite cote serveur.
"""

from flwr.app import Context, Message
from flwr.clientapp import ClientApp

from fl_common.client_helpers import evaluate_client, train_client

app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    return train_client(msg, context, algo="fednova")


@app.evaluate()
def evaluate(msg: Message, context: Context):
    return evaluate_client(msg, context)
