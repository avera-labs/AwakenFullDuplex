import torch

from awaken_chat.inference.sampling import apply_repetition_penalty, class_mask, sample_top_k


def test_temperature_zero_is_argmax():
    logits = torch.tensor([[0.1, 3.0, -1.0]])
    assert int(sample_top_k(logits, 0.0, 5)) == 1


def test_top_k_restricts_support():
    torch.manual_seed(0)
    logits = torch.tensor([[10.0, 9.0, -50.0, -50.0]])
    draws = {int(sample_top_k(logits, 1.0, 2)) for _ in range(50)}
    assert draws <= {0, 1}


def test_class_mask():
    m = class_mask([1, 3, 99], vocab=5, device=torch.device("cpu"))
    assert m.tolist() == [False, True, False, True, False]


def test_repetition_penalty_divides_positive_and_multiplies_negative():
    logits = torch.tensor([[2.0, -2.0, 1.0]])
    out = apply_repetition_penalty(logits, [0, 1], 2.0)
    assert out.tolist() == [[1.0, -4.0, 1.0]]
    assert apply_repetition_penalty(logits, [0], 1.0) is logits
