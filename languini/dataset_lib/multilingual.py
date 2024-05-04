"""Datasets for multilingual data that follow the same interface as LanguiniDatasetIterator"""


import random
import torch
from languini.dataset_lib.languini_books import LanguiniDatasetIterator


class ClonedLanguageDataset(LanguiniDatasetIterator):
    def __init__(self, *, num_languages, p_clone, frac_clone, sp, **kwargs):
        """
        Initialise the dataset iterator. 

        Args:
            same as LanguiniDatasetIterator. Additionally:
            num_languages (int): number of languages to clone the dataset into.
            p_clone (float): probability of using a cloned language for a sample.
            frac_clone (float): fraction of the vocabulary to clone.
            sp: SentencePiece tokenizer.
        """
        super().__init__(**kwargs)
        self.num_languages = num_languages
        self.p_clone = p_clone
        self.frac_clone = frac_clone
        self.sp = sp
        self.original_vocab_size = self.sp.vocab_size()

        # randomly select a fraction of subword ids that are cloned
        clone_seed = 0
        self.n_cloned = int(frac_clone * self.original_vocab_size)
        self.is_cloned = torch.zeros(self.original_vocab_size, dtype=torch.bool)
        all_ids = list(range(self.original_vocab_size))
        random.Random(clone_seed).shuffle(all_ids)
        for i in all_ids[:self.n_cloned]:
            self.is_cloned[i] = True

    def __next__(self):
        batch_x, batch_y, is_padded = self.ds.__next__(raw_seq=True)

        device = batch_x.device
        micro_batches, micro_bsz, seqlen = batch_x.shape
        bsz = micro_batches * micro_bsz

        # sample which cloned language to use for each sample
        cloned_lang = torch.randint(1, self.num_languages, (bsz, 1), device=device)
        do_clone = torch.rand(bsz, 1, dtype=torch.float, device=device) < self.p_l2
        lang = torch.where(do_clone, cloned_lang, 0)
        # lang i has ids [i * size, (i+1) * size)
        lang_offset = lang * self.original_vocab_size

        # map to vocabulary of the "language", don't map padding (0), only map cloned subwords
        lang_offset = lang_offset.view(micro_batches, micro_bsz, 1)
        batch_x = torch.where((batch_x > 0) & self.is_cloned[batch_x], batch_x + lang_offset, batch_x)
        batch_y = torch.where((batch_y > 0) & self.is_cloned[batch_y], batch_y + lang_offset, batch_y)

        return batch_x, batch_y, is_padded

    @property
    def vocab_size(self):
        return self.original_vocab_size * self.num_languages

    def decode(self, ids):
        assert ids.ndim == 1
        # map to original vocab
        ids = ids % self.original_vocab_size
        return self.sp.decode(ids.cpu().tolist())


class BilingualDataset:
    def __init__(
            self,
            *,
            data_path_1,
            split_1,
            data_path_2,
            split_2,
            sp_1,
            sp_2,
            merge_vocab,
            p_l2,
            **kwargs
        ):
        """
        Initialise the dataset iterator. 

        Args:
            same as LanguiniDatasetIterator. Additionally:
            data_path_1 (str): path to the first dataset.
            data_path_2 (str): path to the second dataset.
            split_1 (str): split of the first dataset.
            split_2 (str): split of the second dataset.
            sp_1: SentencePiece tokenizer for the first language.
            sp_2: SentencePiece tokenizer for the second language.
            merge_vocab (bool): whether to merge the vocabularies.
            p_l2 (float): probability of using the second language for a sample.
        """
        self.ds1 = LanguiniDatasetIterator(data_path=data_path_1, split=split_1, **kwargs)
        self.ds2 = LanguiniDatasetIterator(data_path=data_path_2, split=split_2, **kwargs)
        self.sp_1 = sp_1
        self.sp_2 = sp_2
        self.merge_vocab = merge_vocab
        self.p_l2 = p_l2

        # TODO setup vocabularies
    
    def reset(self):
        self.ds1.reset()
        self.ds2.reset()
    
    def __iter__(self):
        return self

    def __next__(self):
        pass

    @property
    def vocab_size(self):
        pass

    def decode(self, ids):
        pass