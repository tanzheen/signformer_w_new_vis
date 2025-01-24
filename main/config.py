config = {
    'data': {
        'img_path': 'path/to/images',
        'max_length': 300,
    },
    'training': {
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'num_epochs': 100,
        'batch_size': 32,
        'max_grad_norm': 1.0,
        'early_stopping_patience': 5,
        'checkpoint_dir': 'checkpoints/',
        'tokens': True  # Based on your dataset code
    },
    'model': {
        # Add model specific configurations
        'hidden_size': 512,
        'num_attention_heads': 8,
        'num_encoder_layers': 6,
        'num_decoder_layers': 6,
        'dropout': 0.1,
    }
} 