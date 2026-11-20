import os
import argparse
import json
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np

from t5_utils import initialize_model, initialize_optimizer_and_scheduler, save_model, load_model_from_checkpoint, setup_wandb
from load_data import load_t5_data, get_tokenizer
from utils import compute_metrics, save_queries_and_records, set_random_seeds

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
PAD_IDX = 0

def get_args():
    '''
    Arguments for training. You may choose to change or extend these as you see fit.
    '''
    parser = argparse.ArgumentParser(description='T5 training loop')

    # Model hyperparameters
    parser.add_argument('--finetune', action='store_true', help="Whether to finetune T5 or not")
    
    # Training hyperparameters
    parser.add_argument('--optimizer_type', type=str, default="AdamW", choices=["AdamW"],
                        help="What optimizer to use")
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0)

    parser.add_argument('--scheduler_type', type=str, default="linear", choices=["none", "cosine", "linear"],
                        help="Whether to use a LR scheduler and what type to use if so")
    parser.add_argument('--num_warmup_epochs', type=int, default=1,
                        help="How many epochs to warm up the learning rate for if using a scheduler")
    parser.add_argument('--max_n_epochs', type=int, default=20,
                        help="How many epochs to train the model for")
    parser.add_argument('--patience_epochs', type=int, default=4,
                        help="If validation performance stops improving, how many epochs should we wait before stopping?")

    parser.add_argument('--use_wandb', action='store_true',
                        help="If set, we will use wandb to keep track of experiments")
    parser.add_argument('--experiment_name', type=str, default='experiment',
                        help="How should we name this experiment?")

    # Data hyperparameters
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--eval_only', action='store_true',
                        help='Load the best checkpoint without training again')
    parser.add_argument('--predict_test', action='store_true',
                        help='Generate test submissions after selecting a configuration on dev')

    args = parser.parse_args()
    if not args.finetune:
        parser.error('Pass --finetune; the optional from-scratch task is not implemented.')
    if min(args.batch_size, args.test_batch_size, args.num_beams, args.max_new_tokens) < 1:
        parser.error('Batch sizes and generation limits must be positive')
    if not args.eval_only and (args.max_n_epochs < 1 or args.patience_epochs < 1):
        parser.error('Training requires positive max_n_epochs and patience_epochs')
    if not args.eval_only and not 0 <= args.num_warmup_epochs < args.max_n_epochs:
        parser.error('Warmup epochs must be nonnegative and less than max_n_epochs')
    return args

def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    best_f1 = -1
    epochs_since_improvement = 0

    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('records', exist_ok=True)
    with open(os.path.join(checkpoint_dir, 'run_config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    get_tokenizer().save_pretrained(checkpoint_dir)
    history = []
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{args.experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{args.experiment_name}_dev.pkl')
    for epoch in range(args.max_n_epochs):
        started = time.monotonic()
        tr_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        train_seconds = time.monotonic() - started
        print(f"Epoch {epoch}: Average train loss was {tr_loss}")
        # Save trained weights before the potentially slower SQL evaluation.
        save_model(checkpoint_dir, model, best=False)

        eval_started = time.monotonic()
        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(args, model, dev_loader,
                                                                         gt_sql_path, model_sql_path,
                                                                         gt_record_path, model_record_path)
        print(f"Epoch {epoch}: Dev loss: {eval_loss}, Record F1: {record_f1}, Record EM: {record_em}, SQL EM: {sql_em}")
        print(f"Epoch {epoch}: {error_rate*100:.2f}% of the generated outputs led to SQL errors")
        eval_seconds = time.monotonic() - eval_started
        print(f"Epoch {epoch}: train {train_seconds:.1f}s, dev {eval_seconds:.1f}s", flush=True)

        if args.use_wandb:
            import wandb
            result_dict = {
                'train/loss' : tr_loss,
                'dev/loss' : eval_loss,
                'dev/record_f1' : record_f1,
                'dev/record_em' : record_em,
                'dev/sql_em' : sql_em,
                'dev/error_rate' : error_rate,
            }
            wandb.log(result_dict, step=epoch)

        if record_f1 > best_f1:
            best_f1 = record_f1
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement == 0:
            save_model(checkpoint_dir, model, best=True)
        history.append(dict(epoch=epoch, train_loss=tr_loss, dev_loss=eval_loss,
                            record_f1=record_f1, record_em=record_em, sql_em=sql_em,
                            error_rate=error_rate, train_seconds=train_seconds,
                            dev_seconds=eval_seconds, best_f1=best_f1,
                            epoch_seconds=time.monotonic() - started))
        print(f"Epoch {epoch}: total including checkpoints {history[-1]['epoch_seconds']:.1f}s", flush=True)
        with open(os.path.join(checkpoint_dir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)

        if epochs_since_improvement >= args.patience_epochs:
            print(f'Early stopping; best dev Record F1: {best_f1:.4f}')
            break

def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss()

    for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(train_loader):
        optimizer.zero_grad()
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        logits = model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            decoder_input_ids=decoder_input,
            use_cache=False,
        )['logits']

        non_pad = decoder_targets != PAD_IDX
        loss = criterion(logits[non_pad], decoder_targets[non_pad])
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None: 
            scheduler.step()

        with torch.no_grad():
            num_tokens = torch.sum(non_pad).item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    return total_loss / total_tokens
        
@torch.no_grad()
def eval_epoch(args, model, dev_loader, gt_sql_pth, model_sql_path, gt_record_path, model_record_path):
    '''
    You must implement the evaluation loop to be using during training. We recommend keeping track
    of the model loss on the SQL queries, the metrics compute_metrics returns (save_queries_and_records should be helpful)
    and the model's syntax error rate. 

    To compute non-loss metrics, you will need to perform generation with the model. Greedy decoding or beam search
    should both provide good results. If you find that this component of evaluation takes too long with your compute,
    we found the cross-entropy loss (in the evaluation set) to be well (albeit imperfectly) correlated with F1 performance.
    '''
    model.eval()
    total_loss, total_tokens = 0.0, 0
    predictions = []
    criterion = nn.CrossEntropyLoss(reduction='sum')
    for source, mask, decoder_input, targets, initial in tqdm(dev_loader, desc='Dev generation'):
        source, mask = source.to(DEVICE), mask.to(DEVICE)
        decoder_input, targets = decoder_input.to(DEVICE), targets.to(DEVICE)
        logits = model(input_ids=source, attention_mask=mask,
                       decoder_input_ids=decoder_input, use_cache=False).logits
        non_pad = targets.ne(PAD_IDX)
        total_loss += criterion(logits[non_pad], targets[non_pad]).item()
        total_tokens += non_pad.sum().item()
        del logits
        # Generation sees only the question and a start token, never gold SQL.
        predictions.extend(generate_queries(args, model, source, mask, initial.to(DEVICE)))

    save_queries_and_records(predictions, model_sql_path, model_record_path)
    sql_em, record_em, record_f1, errors = compute_metrics(
        gt_sql_pth, model_sql_path, gt_record_path, model_record_path)
    error_rate = sum(bool(error) for error in errors) / len(errors)
    return total_loss / total_tokens, float(record_f1), record_em, sql_em, error_rate


def generate_queries(args, model, source, mask, initial):
    generated = model.generate(input_ids=source, attention_mask=mask,
                               decoder_input_ids=initial, do_sample=False,
                               num_beams=args.num_beams, max_new_tokens=args.max_new_tokens)
    return [query.replace('\n', ' ').replace('\r', ' ').strip()
            for query in get_tokenizer().batch_decode(
                generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)]
        
@torch.no_grad()
def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    '''
    You must implement inference to compute your model's generated SQL queries and its associated 
    database records. Implementation should be very similar to eval_epoch.
    '''
    model.eval()
    predictions = []
    for source, mask, initial in tqdm(test_loader, desc='Test generation'):
        predictions.extend(generate_queries(args, model, source.to(DEVICE),
                                            mask.to(DEVICE), initial.to(DEVICE)))
    save_queries_and_records(predictions, model_sql_path, model_record_path)
    print(f'Saved {len(predictions)} test queries to {model_sql_path}')

def main():
    # Get key arguments
    args = get_args()
    set_random_seeds(args.seed)
    if args.use_wandb:
        # Recommended: Using wandb (or tensorboard) for result logging can make experimentation easier
        setup_wandb(args)

    # Load the data and the model
    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)
    print(f'Device: {DEVICE}; train/dev/test examples: '
          f'{len(train_loader.dataset)}/{len(dev_loader.dataset)}/{len(test_loader.dataset)}')
    print(f'Train batches per epoch: {len(train_loader)}')

    # Train 
    if not args.eval_only:
        model = initialize_model(args)
        optimizer, scheduler = initialize_optimizer_and_scheduler(args, model, len(train_loader))
        train(args, model, train_loader, dev_loader, optimizer, scheduler)
        del model, optimizer, scheduler
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

    # Evaluate
    model = load_model_from_checkpoint(args, best=True)
    model.eval()
    
    # Dev set
    experiment_name = args.experiment_name
    model_type = 'ft' if args.finetune else 'scr'
    gt_sql_path = os.path.join(f'data/dev.sql')
    gt_record_path = os.path.join(f'records/ground_truth_dev.pkl')
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_dev.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_dev.pkl')
    dev_loss, dev_record_f1, dev_record_em, dev_sql_em, dev_error_rate = eval_epoch(args, model, dev_loader,
                                                                                    gt_sql_path, model_sql_path,
                                                                                    gt_record_path, model_record_path)
    print(f"Dev set results: Loss: {dev_loss}, Record F1: {dev_record_f1}, Record EM: {dev_record_em}, SQL EM: {dev_sql_em}")
    print(f"Dev set results: {dev_error_rate*100:.2f}% of the generated outputs led to SQL errors")

    # Test set
    model_sql_path = os.path.join(f'results/t5_{model_type}_{experiment_name}_test.sql')
    model_record_path = os.path.join(f'records/t5_{model_type}_{experiment_name}_test.pkl')
    if args.predict_test:
        test_inference(args, model, test_loader, model_sql_path, model_record_path)
    else:
        print('Test generation skipped. After choosing the configuration on dev, '
              'use --eval_only --predict_test with the same experiment_name.')

if __name__ == "__main__":
    main()
