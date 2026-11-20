import os
from functools import lru_cache

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import T5TokenizerFast, AddedToken
import torch

PAD_IDX = 0
MODEL_NAME = 'google-t5/t5-small'
INPUT_PREFIX = 'translate English to SQL: '


@lru_cache(maxsize=1)
def get_tokenizer():
    tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
    # Stock T5 maps '<' to UNK. Keep the complete operators and preceding
    # separator so decoding also preserves '<=' and exact SQL whitespace.
    tokenizer.add_tokens([AddedToken(' <', normalized=False),
                          AddedToken(' <=', normalized=False)])
    return tokenizer

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        if split not in {'train', 'dev', 'test'}:
            raise ValueError(f'Unknown split: {split}')
        self.split = split
        self.tokenizer = get_tokenizer()
        self.examples = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        texts = load_lines(os.path.join(data_folder, f'{split}.nl'))
        # Preserve SQL case, quotes and identifiers; no target truncation.
        encoder_ids = tokenizer([INPUT_PREFIX + text for text in texts],
                                truncation=False)['input_ids']
        if split == 'test':
            return [(torch.tensor(ids, dtype=torch.long),) for ids in encoder_ids]
        sql = load_lines(os.path.join(data_folder, f'{split}.sql'))
        if len(texts) != len(sql):
            raise ValueError(f'{split}: natural-language/SQL line counts differ')
        target_ids = tokenizer(sql, truncation=False)['input_ids']
        return [(torch.tensor(source, dtype=torch.long),
                 torch.tensor(target, dtype=torch.long))
                for source, target in zip(encoder_ids, target_ids)]
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids = pad_sequence([item[0] for item in batch], batch_first=True,
                               padding_value=PAD_IDX)
    encoder_mask = encoder_ids.ne(PAD_IDX).long()
    targets = [item[1] for item in batch]  # Includes the tokenizer's EOS token.
    decoder_targets = pad_sequence(targets, batch_first=True, padding_value=PAD_IDX)
    # T5's pretrained decoder starts with PAD (0). Teacher forcing shifts by one.
    decoder_inputs = pad_sequence(
        [torch.cat((target.new_tensor([PAD_IDX]), target[:-1])) for target in targets],
        batch_first=True, padding_value=PAD_IDX)
    initial_decoder_inputs = torch.full((len(batch), 1), PAD_IDX, dtype=torch.long)
    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids = pad_sequence([item[0] for item in batch], batch_first=True,
                               padding_value=PAD_IDX)
    encoder_mask = encoder_ids.ne(PAD_IDX).long()
    initial_decoder_inputs = torch.full((len(batch), 1), PAD_IDX, dtype=torch.long)
    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # TODO
    return train_x, train_y, dev_x, dev_y, test_x
