"""Client Flower pour FedProx.

La seule difference avec FedAvg est le coefficient proximal `mu`, envoye
par le serveur dans la configuration du round.
"""

from flwr.app import Context, Message
from flwr.clientapp import ClientApp

from fl_common.client_helpers import evaluate_client, train_client

app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    return train_client(msg, context, algo="fedprox")


@app.evaluate()
def evaluate(msg: Message, context: Context):
    return evaluate_client(msg, context)
