import unittest
import numpy as np
import torch
from chronos.chronos2.config import Chronos2CoreConfig
from chronos.chronos2.model import Chronos2Model
from src.c2_lora_tuning import attach_lora, reset_lora, configure_lora, adapter_state, load_adapter_state, frozen_state_fingerprint


class C2LoraTests(unittest.TestCase):
    def test_only_requested_adapters_update_and_disabling_recovers_foundation(self):
        for mode in ("time", "time_head"):
            with self.subTest(mode=mode):
                base = Chronos2Model(Chronos2CoreConfig(d_model=16, d_ff=32, d_kv=8, num_heads=2,
                    num_layers=2, chronos_config=dict(context_length=64, input_patch_size=16,
                    input_patch_stride=16, output_patch_size=16,
                    quantiles=[i/10 for i in range(1,10)], use_reg_token=True)))
                model, targets = attach_lora(base, rank=2)
                self.assertEqual(len(targets), 11)
                reset_lora(model, 42)
                names = configure_lora(model, mode)
                original = adapter_state(model)
                fingerprint = frozen_state_fingerprint(model)
                context = torch.tensor(np.linspace(1., .85, 128).reshape(2,64), dtype=torch.float32)
                future = context[:, -1:] - torch.arange(1,17)[None,:] * .001
                batch = dict(context=context, future_target=future, group_ids=torch.arange(2))
                with torch.no_grad():
                    before = model(**batch).quantile_preds.clone()
                optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01, weight_decay=0.)
                model(**batch).loss.backward()
                optimizer.step()
                changed = {n for n, t in adapter_state(model).items() if not torch.equal(original[n], t)}
                self.assertTrue(changed)
                self.assertTrue(changed.issubset(names))
                self.assertTrue(any(".layer.0.self_attention." in n for n in changed))
                self.assertEqual(any("output_patch_embedding" in n for n in changed), mode=="time_head")
                self.assertEqual(fingerprint, frozen_state_fingerprint(model))
                with model.disable_adapter(), torch.no_grad():
                    torch.testing.assert_close(before, model(**batch).quantile_preds)
                load_adapter_state(model, original)
                with torch.no_grad():
                    torch.testing.assert_close(before, model(**batch).quantile_preds)


if __name__ == "__main__":
    unittest.main()
