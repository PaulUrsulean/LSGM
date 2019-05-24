import argparse
from src.trainer import Trainer, generate_config
from src.model.modules import MLPEncoder, RNNDecoder


parser = argparse.ArgumentParser()

# Data related options
parser.add_argument('--n-atoms', type=int, default=5,
                    help='Number of atoms in simulation.')
parser.add_argument('--n-edges', type=int, default=2,
                    help='Number of edge types to infer.')
parser.add_argument('--n-features', type=int, default=2,
                    help='The number of features per time step per node')
parser.add_argument('--n-timesteps', type=int, default=49,
                    help='The number of time steps per sample.')
parser.add_argument('--prediction-steps', type=int, default=20,
                    help='Num steps to predict before re-using teacher forcing.')

# Batching/Training options
parser.add_argument('--batch-size', type=int, default=128,
                    help='Number of samples per batch.')
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.')
parser.add_argument('--use-early-stopping', action='store_true',
                    help='Whether to stop when val loss stagnates')
parser.add_argument('--patience', type=int, default=1,
                    help='Number of epochs to wait without progress before stopping if early_stop active.')

# Optimization parameters
parser.add_argument('--lr', type=float, default=0.0005,
                    help='Initial learning rate.')
parser.add_argument('--lr-decay-freq', type=int, default=200,
                    help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='LR decay factor.')
parser.add_argument('--temp', type=float, default=0.5,
                    help='Temperature for Gumbel softmax.')
parser.add_argument('--var', type=float, default=5e-5,
                    help='Output variance.')
parser.add_argument('--hard', action='store_true',
                    help='Uses discrete samples in training forward pass.')
parser.add_argument('--burn-in', action='store_true',
                    help='Whether to use a burn-in phase (RNN only)')

# Neural Network parameters
parser.add_argument('--encoder', type=str, default='mlp',
                    help='Type of path encoder model (mlp or cnn).')
parser.add_argument('--decoder', type=str, default='mlp',
                    help='Type of decoder model (mlp, rnn, or sim).')
parser.add_argument('--encoder-hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--decoder-hidden', type=int, default=256,
                    help='Number of hidden units.')
parser.add_argument('--encoder-dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--decoder-dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')

# Logging options
parser.add_argument('--save-folder', type=str, default='logs',
                    help='Where to save the trained model, leave empty to not save anything.')
parser.add_argument('--load-folder', type=str, default='',
                    help='Where to load the trained model if finetunning. ' +
                         'Leave empty to train from scratch')
parser.add_argument('--log-freq', type=int, default=10,
                    help='Logging frequency')
parser.add_argument('--store-model', action='store_true',
                    help='Whether to store resulting model in the specified save_folder')

# Other
parser.add_argument('--gpu-id', type=int, default=None,
                    help='Specifies which GPU to train on. If left empty, the CPU is used.')

# Possibly implement
parser.add_argument('--skip-first', action='store_true', default=False,
                    help='Skip first edge type in decoder, i.e. it represents no-edge.')
parser.add_argument('--prior', action='store_true', default=False,
                    help='Whether to use sparsity prior.')
parser.add_argument('--dynamic-graph', action='store_true', default=False,
                    help='Whether test with dynamically re-computed graph.')

args = parser.parse_args()

config = generate_config(
    n_edges = args.n_edges,
    n_atoms = args.n_atoms,
    epochs = args.epochs,
    use_early_stopping = args.use_early_stopping,
    early_stopping_patience = args.patience,
    temp = args.temp,
    gpu_id = args.gpu_id,
    
    timesteps = args.n_timesteps,
    prediction_steps = args.prediction_steps,
    pred_steps = args.prediction_steps, # Duplicate, check issue #24

    hard = args.hard,
    burn_in = args.burn_in,

    log_step = args.log_freq,
    log_dir = args.save_folder,
    logger_config = "",  # str ???
    store_models = args.store_model,


    scheduler_stepsize = args.lr_decay_freq,
    scheduler_gamma = args.gamma,  # Decay rate of learning rate

    adam_learning_rate = args.lr,  # normally 1e-3

    prior=None,
    add_const=False,
    eps=1e-16,
    beta=1.0,
    prediction_variance=args.var
)

encoder = MLPEncoder(args.n_timesteps * args.n_features, args.encoder_hidden, args.n_edges)
decoder = RNNDecoder(n_in_node=args.n_features, edge_types=args.n_edges, n_hid=args.decoder_hidden)

# TODO: DATA LOADERS
# trainer = Trainer(encoder=encoder,
#                 decoder=decoder,
#                 data_loaders=data_loaders,
#                 config=config)
# trainer.train()

# TODO: EVALUATION - WAIT UNTIL EVALUATOR IS REFACTORED INTO TRAINER CLASS