_default_config = dict(

    globals=dict(
        seed=42,
        prior=None,
        add_const=False,
        eps=1e-16
    ),

    training=dict(
        gpu_id=None,
        use_early_stopping=False,
        early_stopping_mode='min',  # in ["min", "max"]
        early_stopping_metric='val_mse_loss',
        early_stopping_patience=20,
        epochs=500,
        batch_size=128,

        optimizer=dict(
            type='adam',
            learning_rate=0.0005,
            betas=(0.9, 0.999)
        ),

        scheduler=dict(
            stepsize=200,
            gamma=0.5,
        )
    ),

    data=dict(
        timesteps=49,
        type='springs',
        springs=dict(
            suffix='_springs5',
            dims=4
        )
    ),

    loss=dict(
        beta=1.0
    ),

    model=dict(
        save=True,
        # In ground truth
        prediction_steps=10,
        factor_graph=True,
        skip_first=False,
        hard=False,
        dynamic_graph=False,
        temp=0.5,
        burn_in=False,
        n_edge_types=2,
        encoder=dict(
            model='mlp',  # or CNN
            hidden_dim=256,
            dropout=0.5
        ),
        decoder=dict(
            model='rnn',  # or MLP
            hidden_dim=256,
            dropout=0.0,
            prediction_variance=5e-5
        )),

    logging=dict(
        log_step=10,
        log_dir='./logs',
        logger_config="",  # str
        store_models=True
    )
)


def generate_config(n_atoms, n_edges=2, *args, **kwargs):
    config = _default_config.copy()

    config['model']['n_edge_types'] = n_edges
    config['data']['n_atoms'] = n_atoms

    # Override other parameterswith manually set values
    for key, value in kwargs.items():
        # assert (key in config)
        config[key] = value
    return config
