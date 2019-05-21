_default_config = dict(
    epochs=30,

    use_early_stopping=False,
    early_stopping_patience=1,
    early_stopping_mode='min',  # in ["min", "max"]
    early_stopping_metric='val_loss',

    gpu_id=None,  # or None


    timesteps=1,  # In forecast
    prediction_steps=2,  #

    temp=0.5,
    hard=False,
    burn_in=False,

    log_step=10,
    log_dir='./logs',
    logger_config="",  # str
    store_models=True,

    pred_steps=1,

    scheduler_stepsize=200,
    scheduler_gamma=0.5,  # Decay rate of learning rate

    adam_learning_rate=0.0005,  # normally 1e-3
    adam_betas=(0.9, 0.999),

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
