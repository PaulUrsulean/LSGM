_default_config = dict(
    epochs=500,

    globals=dict(
        seed=42
    ),

    training=dict(
        batch_size=128
    ),

    data=dict(
        name='springs',
        springs=dict(
            suffix='_springs5',
            dims=4
        )
    ),

    adam_learning_rate=0.0005,
    adam_betas=(0.9, 0.999),

    model=dict(
        save=True,
        factor_graph=True,
        skip_first=False,
        hard=False,
        dynamic_graph=False,
        encoder=dict(
            model='mlp',  # or CNN
            hidden_dim=256,
            dropout=0.5
        ),
        decoder=dict(
            model='rnn',  # or MLP
            hidden_dim=256,
            dropout=0.0
        )),

    temp=0.5,
    hard=False,
    burn_in=False,

    gpu_id=None,

    use_early_stopping=False,
    early_stopping_patience=20,
    early_stopping_mode='min',  # in ["min", "max"]
    early_stopping_metric='val_mse_loss',

    # or None

    # Specifically, we run the encoder on the first 49 time
    # steps (same as in training and validation), then predict with
    # our decoder the following 20 unseen time steps.
    timesteps=49,  # In ground truth
    prediction_steps=10,

    log_step=10,
    log_dir='./logs',
    logger_config="",  # str
    store_models=True,

    scheduler_stepsize=200,
    scheduler_gamma=0.5,  # Decay rate of learning rate

    prior=None,
    add_const=False,
    eps=1e-16,
    beta=1.0,
    prediction_variance=5e-5
)


def generate_config(n_edges, n_atoms, *args, **kwargs):
    config = _default_config.copy()

    config['n_edge_types'] = n_edges
    config['n_atoms'] = n_atoms

    # Override other parameterswith manually set values
    for key, value in kwargs.items():
        config[key] = value
    return config
