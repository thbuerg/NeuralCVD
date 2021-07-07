import torch.nn as nn


class ShallowMLP(nn.Module):
    def __init__(self, input_dim=32, output_dim=2, activation=None, final_batchnorm=False, dropout=0.5, **kwargs):
        super(ShallowMLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        if activation is not None and isinstance(activation, str):
            m = activation.split('.')
            activation = getattr(nn, m[1])
            print(activation)
        print(self.output_dim)

        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(self.dropout),
            nn.SELU(True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(self.dropout),
            nn.SELU(True),
        )

        predictor_specs = [nn.Linear(128, self.output_dim), ]
        if final_batchnorm:
            predictor_specs.append(nn.BatchNorm1d(self.output_dim))
        if activation is not None:
            predictor_specs.append(activation())
        self.predictor = nn.Sequential(*predictor_specs)

    def forward(self, input):
        fts = self.mlp(input)
        fts = self.predictor(fts)
        return fts


class StandardMLP(nn.Module):
    def __init__(self, input_dim=32, output_dim=2, activation=None, final_batchnorm=False, dropout=0.5, **kwargs):
        """
        A simple feed-forward neural network.

        :param input_dim:   `int`, dimension ot the input features
        :param output_dim:  `int`, dimension of the outlayer
        :param activation:  `nn.Module`, NOT initialized. that is the activation of the last layer, if `None` no activation will be performed.
        :param dropout:     `float`, [<1], that specifies the dropout probability
        :param kwargs:
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        if activation is not None and isinstance(activation, str):
            m = activation.split('.')
            activation = getattr(nn, m[1])
            print(activation)
        self.activation = activation
        self.mlp = nn.Sequential(

            nn.Linear(input_dim, 256),
            nn.Dropout(self.dropout),
            nn.BatchNorm1d(256),
            nn.ReLU6(True),

            nn.Linear(256, 256),
            nn.Dropout(self.dropout),
            nn.SELU(True),

            nn.Linear(256, 256),
            nn.Dropout(self.dropout),
            nn.SELU(True),
        )

        predictor_specs = [nn.Linear(256, self.output_dim),
                           ]
        if final_batchnorm:
            predictor_specs.append(nn.BatchNorm1d(self.output_dim))
        if activation is not None:
            predictor_specs.append(activation())

        self.predictor = nn.Sequential(*predictor_specs)

    def forward(self, input):
        fts = self.mlp(input)
        output = self.predictor(fts)
        return output

