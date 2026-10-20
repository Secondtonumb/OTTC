"""Independent transport, repeated-target, gradient and text-free decode tests."""
import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from .head import (
    OTTCTransducerHead, monotone_transport, mass_boundaries, upstream_cost, all_phone_costs,
)
from .model import wvOTTCTransducerConfig, wvForOTTCTransducer
from .archived_ot_loss import (
    batched_ottc_loss_bucketized, jensen_shannon_divergence,
)


def head(**kwargs):
    values = dict(input_dim=4, phone_ids=[1,2,3], external_vocab_size=5,
        embedding_dim=8, predictor_dim=8, predictor_layers=1, joint_dim=8, dropout=0)
    return OTTCTransducerHead(**(values | kwargs))


def dense_reference(mass, lengths):
    """Independent interval-overlap construction, rather than bucketization."""
    a_end = mass.cumsum(-1)
    a_start = a_end-mass
    bins = torch.arange(int(lengths.max()), device=mass.device, dtype=mass.dtype)
    b_start = bins[None]/lengths[:,None]
    b_end = (bins[None]+1)/lengths[:,None]
    overlap = (torch.minimum(a_end[:,:,None],b_end[:,None])-torch.maximum(a_start[:,:,None],b_start[:,None])).clamp_min(0)
    return overlap.masked_fill(bins[None,None]>=lengths[:,None,None],0)


class ModelTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)

    def test_sparse_transport_against_dense_reference_and_marginals(self):
        logits = torch.randn(2,7,dtype=torch.float64,requires_grad=True)
        valid = torch.arange(7)[None]<torch.tensor([7,4])[:,None]
        mass = logits.masked_fill(~valid,-torch.inf).softmax(-1)
        lengths = torch.tensor([3,6])  # Includes U>T and nonuniform valid lengths.
        sparse = monotone_transport(mass,valid,lengths)
        dense = dense_reference(mass,lengths)
        rebuilt = torch.zeros_like(dense)
        rebuilt.index_put_((sparse.batch,sparse.frame,sparse.occurrence),sparse.weight,accumulate=True)
        torch.testing.assert_close(rebuilt,dense)
        torch.testing.assert_close(rebuilt.sum(-1),mass)
        torch.testing.assert_close(rebuilt.sum(1),
            (torch.arange(6)[None]<lengths[:,None]).double()/lengths[:,None])
        cost = torch.randn_like(dense)
        lhs = (sparse.weight*cost[sparse.batch,sparse.frame,sparse.occurrence]).sum()
        rhs = (dense*cost).sum()
        gl = torch.autograd.grad(lhs,logits,retain_graph=True)[0]
        gr = torch.autograd.grad(rhs,logits)[0]
        torch.testing.assert_close(gl,gr)

    def test_costs_reuse_archived_formula(self):
        logits = torch.randn(4,3,dtype=torch.float64)
        targets = torch.tensor([0,0,1,2])
        expected = jensen_shannon_divergence(logits.softmax(-1),F.one_hot(targets,3).double()).sum(-1)
        torch.testing.assert_close(upstream_cost(logits,targets),expected)
        torch.testing.assert_close(all_phone_costs(logits)[torch.arange(4),targets],expected)

    def test_sparse_joint_loss_matches_dense_joint_loss(self):
        model = head().double().eval()
        features = torch.randn(2,5,4,dtype=torch.float64)
        valid = torch.tensor([[1,1,1,1,1],[1,1,1,0,0]],dtype=torch.bool)
        labels = torch.tensor([[1,1,2],[3,3,-100]])
        output = model(features,valid,labels)
        acoustic,mass,_,_ = model.acoustic(features,valid)
        tokens,lengths = model.targets(labels)
        history,_ = model.predictor(F.pad(tokens[:,:-1],(1,0),value=model.predictor.bos_id))
        gamma = dense_reference(mass,lengths)
        expected = features.new_zeros(())
        for b in range(2):
            for u in range(int(lengths[b])):
                for t in range(int(valid[b].sum())):
                    z = model.joint_logits(acoustic[b,t][None],history[b,u][None])
                    expected = expected+gamma[b,t,u]*upstream_cost(z,tokens[b,u][None])[0]
        torch.testing.assert_close(output['ottc_sum'],expected)
        sparse_grad = torch.autograd.grad(output['ottc_sum'],model.mass_head.weight,retain_graph=True)[0]
        dense_grad = torch.autograd.grad(expected,model.mass_head.weight)[0]
        torch.testing.assert_close(sparse_grad,dense_grad,atol=1e-10,rtol=1e-8)

    def test_repeated_occurrences_have_distinct_bins(self):
        model = head()
        labels = torch.tensor([[1,1,1,2]])
        out = model(torch.randn(1,8,4),torch.ones(1,8,dtype=torch.bool),labels)
        transport = out['transport']
        self.assertEqual(set(transport.occurrence.tolist()),{0,1,2,3})
        for u in range(4):
            torch.testing.assert_close(transport.weight[transport.occurrence==u].sum(),torch.tensor(.25))
        self.assertEqual(out['pair_logits'].shape[-1],3)  # Three phones, no blank/BOS/pad/unk output.

    def test_decode_preserves_repeats_and_uses_predicted_count(self):
        model = head().eval()
        with torch.no_grad():
            model.count_head.weight.zero_(); model.count_head.bias.fill_(0.)
            model.output_projection.weight.zero_(); model.output_projection.bias.copy_(torch.tensor([5.,-5.,-5.]))
        features = torch.randn(1,3,4)
        valid = torch.ones(1,3,dtype=torch.bool)
        pred = model.recognize(features,valid)
        self.assertEqual(pred['tokens'],[[1,1,1]])
        self.assertEqual(len(pred['boundary_positions'][0]),2)
        with torch.no_grad(): model.count_head.bias.fill_(torch.log(torch.tensor(2/3)))
        self.assertEqual(model.recognize(features,valid)['tokens'],[[1,1]])

    def test_count_cap_and_lower_bound_are_reported(self):
        model = head(max_output_tokens=4).eval()
        features,mask = torch.randn(1,2,4),torch.ones(1,2,dtype=torch.bool)
        for bias,count in ((20.,4),(-20.,1)):
            with torch.no_grad(): model.count_head.bias.fill_(bias)
            pred = model.recognize(features,mask)
            self.assertEqual(len(pred['tokens'][0]),count)
            self.assertTrue(bool(pred['length_clipped'][0]))

    def test_quantile_boundaries(self):
        torch.testing.assert_close(mass_boundaries(torch.tensor([.25,.75]),3),torch.tensor([1+1/9,1+5/9]))
        self.assertEqual(len(mass_boundaries(torch.tensor([1.]),1)),0)

    def test_future_targets_cannot_leak_into_predictor(self):
        model = head().eval()
        a = torch.tensor([[model.predictor.bos_id,0,0,1]])
        b = torch.tensor([[model.predictor.bos_id,0,2,2]])
        pa,_ = model.predictor(a); pb,_ = model.predictor(b)
        torch.testing.assert_close(pa[:,:2],pb[:,:2])
        self.assertGreater(float((pa[:,2:]-pb[:,2:]).abs().sum()),0.)

    def test_padding_and_batched_decode(self):
        model = head().eval()
        features = torch.randn(2,6,4)
        valid = torch.arange(6)[None]<torch.tensor([6,3])[:,None]
        changed = features.clone(); changed[~valid] = 1e6
        labels = torch.tensor([[1,1,2],[2,2,-100]])
        torch.testing.assert_close(model(features,valid,labels)['loss'],model(changed,valid,labels)['loss'])
        batch = model.recognize(changed,valid)
        for b,length in enumerate((6,3)):
            single = model.recognize(features[b:b+1,:length],valid[b:b+1,:length])
            self.assertEqual(batch['tokens'][b],single['tokens'][0])
            torch.testing.assert_close(batch['boundary_positions'][b],single['boundary_positions'][0])

    def test_all_heads_and_features_receive_finite_gradients(self):
        model = head()
        features = torch.randn(2,7,4,requires_grad=True)
        output = model(features,torch.ones(2,7,dtype=torch.bool),torch.tensor([[1,1,2],[3,2,3]]))
        output['loss'].backward()
        for name in ('mass_head','count_head','predictor','acoustic_projection','output_projection'):
            gradients = [p.grad for p in getattr(model,name).parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients),name)
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients),0.,name)
        self.assertTrue(torch.isfinite(features.grad).all())

    def test_invalid_targets_and_masks(self):
        model = head()
        features,mask = torch.randn(1,3,4),torch.ones(1,3,dtype=torch.bool)
        for labels in (torch.tensor([[0]]),torch.tensor([[4]]),torch.tensor([[1,-100,2]]),torch.tensor([[-100]])):
            with self.assertRaises(ValueError): model(features,mask,labels)
        with self.assertRaises(ValueError): model(features,torch.tensor([[1,0,1]],dtype=torch.bool),torch.tensor([[1]]))

    def test_synthetic_repeat_overfit_without_reference_decode(self):
        model = head()
        # Phone-like acoustic states: two indistinguishable consecutive
        # occurrences of phone1/phone3, followed by phone2. Count must supply
        # the multiplicity even though the first two acoustic states match.
        features = torch.zeros(2,9,4)
        features[0,:6,0] = 1; features[1,:6,2] = 1; features[:,6:,1] = 1
        mask = torch.ones(2,9,dtype=torch.bool)
        labels = torch.tensor([[1,1,2],[3,3,2]])
        optimizer = torch.optim.Adam(model.parameters(),lr=.025)
        initial = float(model(features,mask,labels)['loss'])
        for _ in range(180):
            optimizer.zero_grad()
            out = model(features,mask,labels)
            out['loss'].backward(); optimizer.step()
        model.eval()
        self.assertLess(float(model(features,mask,labels)['loss']),initial*.1)
        self.assertEqual(model.recognize(features,mask)['tokens'],labels.tolist())

    def test_wav2vec2_wrapper_checkpoint_and_text_free_interface(self):
        from .train import tiny_config
        model = wvForOTTCTransducer(wvOTTCTransducerConfig(encoder_config=tiny_config(),
            phone_ids=[1,2,3],external_vocab_size=5,embedding_dim=8,predictor_dim=8,
            predictor_layers=1,joint_dim=8,dropout=0)).eval()
        model.freeze_feature_encoder()
        audio = torch.randn(1,3200)
        mask = torch.ones_like(audio,dtype=torch.long)
        before = model.recognize(audio,mask)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'state.pt'
            torch.save(dict(model=model.state_dict(),config=model.config.to_dict()),path)
            state = torch.load(path,weights_only=False)
            restored = wvForOTTCTransducer(wvOTTCTransducerConfig(**state['config'])).eval()
            restored.load_state_dict(state['model'])
        after = restored.recognize(audio,mask)
        self.assertEqual(before['tokens'],after['tokens'])
        torch.testing.assert_close(before['boundary_positions'][0],after['boundary_positions'][0])
        self.assertEqual(model.phone_head.output_projection.out_features,3)
        self.assertTrue(all(not p.requires_grad for p in model.wav2vec2.feature_extractor.parameters()))

    @unittest.skipUnless(torch.cuda.is_available(),'requires allocated CUDA GPU')
    def test_native_cuda_transport_value_and_gradient(self):
        model = head().cuda().eval()
        features = torch.randn(2,7,4,device='cuda')
        mask = torch.arange(7,device='cuda')[None]<torch.tensor([7,5],device='cuda')[:,None]
        labels = torch.tensor([[1,1,2],[3,3,-100]],device='cuda')
        out = model(features,mask,labels)
        mass = out['mass']
        lengths = torch.tensor([3,2],device='cuda')
        gamma = dense_reference(mass,lengths)
        sparse = out['transport']
        rebuilt = torch.zeros_like(gamma)
        rebuilt.index_put_((sparse.batch,sparse.frame,sparse.occurrence),sparse.weight,accumulate=True)
        torch.testing.assert_close(rebuilt,gamma,atol=2e-7,rtol=2e-6)
        # With history contribution disabled, joint-OTTC must reduce to the
        # archived frame-only upstream objective, including mass gradients.
        with torch.no_grad():
            model.predictor.projection.weight.zero_(); model.predictor.projection.bias.zero_()
        output = model(features,mask,labels)
        acoustic,mass,_,_ = model.acoustic(features,mask)
        logits = model.output_projection(model.joint(acoustic[:,:,None],torch.zeros_like(acoustic[:,:,None])))[:,:,0]
        tokens,_ = model.targets(labels)
        valid_target = torch.arange(3,device='cuda')[None]<lengths[:,None]
        native = batched_ottc_loss_bucketized(logits,F.one_hot(tokens,3).float(),mass,
            valid_target.float()/lengths[:,None],amask=mask,bmask=valid_target,jsd=True)[0]
        torch.testing.assert_close(output['ottc_sum'],native,atol=1e-5,rtol=1e-5)
        left = torch.autograd.grad(output['ottc_sum'],model.mass_head.weight,retain_graph=True)[0]
        right = torch.autograd.grad(native,model.mass_head.weight)[0]
        torch.testing.assert_close(left,right,atol=2e-5,rtol=2e-5)


if __name__ == '__main__':
    unittest.main()
