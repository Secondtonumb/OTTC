"""Meaningful CPU checks for zero quantity loss and batched evaluation."""
import tempfile
import unittest

import torch
from transformers import Wav2Vec2Config

from ottc.models.sequence.wav2vec2_cif import wvCIFConfig,wvForCIF
from reproduction.cif_noquantity_20261016.data import load_rows,collate
from reproduction.cif_noquantity_20261016.evaluate import evaluate


class RecipeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4);torch.manual_seed(42)
        cls.rows,cls.vocab,_=load_rows('buckeye',8,4)
        base=Wav2Vec2Config(hidden_size=32,num_hidden_layers=1,num_attention_heads=2,intermediate_size=64,
            conv_dim=(16,16,16),conv_stride=(10,8,4),conv_kernel=(20,10,5),feat_extract_norm='layer',
            num_conv_pos_embedding_groups=2,num_conv_pos_embeddings=16,mask_time_prob=0,
            hidden_dropout=0,attention_dropout=0,feat_proj_dropout=0,layerdrop=0)
        cls.model=wvForCIF(wvCIFConfig(encoder_config=base.to_dict(),vocab_size=42,decoder_dim=16,
            decoder_heads=2,decoder_layers=1,decoder_ffn=32,dropout=0,quantity_weight=0)).eval()

    def test_no_quantity_in_objective_and_cif_gradient(self):
        self.model.zero_grad(set_to_none=True)
        out=self.model(**collate(self.rows['train'][:4],'cpu'))
        torch.testing.assert_close(out['loss'],out['ce_sum']/out['token_count'])
        out['loss'].backward()
        grad=self.model.phone_head.cif.weight_proj.weight.grad
        self.assertTrue(torch.isfinite(grad).all());self.assertGreater(float(grad.abs().sum()),0)

    def test_batched_predictions_and_loss(self):
        import json
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            a=evaluate(self.model,self.rows['dev'],self.vocab,Path(d)/'single.json',batch_size=1)
            b=evaluate(self.model,self.rows['dev'],self.vocab,Path(d)/'batch.json',batch_size=4)
            aa={x['utt']:x for x in map(json.loads,(Path(d)/'single.predictions.jsonl').read_text().splitlines())}
            bb={x['utt']:x for x in map(json.loads,(Path(d)/'batch.predictions.jsonl').read_text().splitlines())}
            for key in aa:
                self.assertEqual(aa[key]['hypothesis'],bb[key]['hypothesis'])
                torch.testing.assert_close(torch.tensor(aa[key]['boundaries']['text_free']),
                    torch.tensor(bb[key]['boundaries']['text_free']),atol=2e-5,rtol=1e-5)
            self.assertAlmostEqual(a['ce_per_phone'],b['ce_per_phone'],places=5)


if __name__=='__main__':unittest.main()
