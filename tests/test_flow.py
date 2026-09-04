import torch

from harp.config import HarmonicConfig, ModelConfig
from harp.flow import HARPFlow, flow_coefficients
from harp.flow_transform import FlowTransform
from harp.model import HARPCore


def test_analytic_coefficients_at_path_endpoints() -> None:
    variance = torch.tensor([0.25, 1.0, 4.0])
    at_noise = flow_coefficients(torch.tensor([0.0]), variance)
    torch.testing.assert_close(at_noise.c_in[0, 0], torch.ones(3))
    torch.testing.assert_close(at_noise.c_skip[0, 0], -torch.ones(3))
    torch.testing.assert_close(at_noise.c_out[0, 0], variance.sqrt())

    at_data = flow_coefficients(torch.tensor([1.0]), variance)
    torch.testing.assert_close(at_data.c_in[0, 0], variance.rsqrt())
    torch.testing.assert_close(at_data.c_skip[0, 0], torch.ones(3))
    torch.testing.assert_close(at_data.c_out[0, 0], torch.ones(3))


def test_residual_target_has_unit_variance_for_diagonal_gaussian() -> None:
    torch.manual_seed(3)
    count = 300_000
    variance = torch.tensor([0.2, 1.0, 5.0])
    timestep = torch.full((count,), 0.5)
    target = torch.randn(count, 1, 3) * variance.sqrt()
    noise = torch.randn_like(target)
    state = 0.5 * noise + 0.5 * target
    coefficients = flow_coefficients(timestep, variance)
    residual = (target - noise - coefficients.c_skip * state) / coefficients.c_out
    torch.testing.assert_close(residual.mean((0, 1)), torch.zeros(3), atol=0.01, rtol=0)
    torch.testing.assert_close(residual.var((0, 1)), torch.ones(3), atol=0.015, rtol=0)


def test_small_flow_trains_and_samples_in_raw_mel_space() -> None:
    channels = 8
    model = HARPCore(
        ModelConfig(
            mel_channels=channels,
            content_dim=16,
            dim=32,
            depth=2,
            head_dim=8,
            ff_hidden_dim=64,
            kernel_size=5,
            time_code_dim=32,
            speaker_code_dim=32,
            adaln_rank=8,
            adaln_mixer_dim=16,
            harmonic_dim=8,
            harmonic_injection_blocks=(1,),
        ),
        HarmonicConfig(sample_rate=16000, fmin=40, fmax=7600),
        num_speakers=3,
    )
    transform = FlowTransform(
        mean=torch.linspace(-2, 1, channels),
        basis=torch.eye(channels),
        gain=torch.ones(channels),
        lambda_raw=torch.ones(channels),
        lambda_effective=torch.ones(channels),
        metadata={"artifact_type": "flow_transform_v1"},
    )
    system = HARPFlow(model, transform)
    mask = torch.tensor([[True] * 7, [True] * 5 + [False] * 2])
    batch = {
        "mel": torch.randn(2, 7, channels),
        "content": torch.randn(2, 7, 16),
        "f0": torch.rand(2, 7, 1) * 400 + 80,
        "rms": torch.randn(2, 7, 1),
        "speaker": torch.tensor([0, 2]),
        "mask": mask,
    }
    loss = system(batch)
    loss.total.backward()
    assert loss.total.isfinite()
    assert model.output.weight.grad is not None
    generated = system.eval().sample(
        batch["content"][:1],
        batch["f0"][:1],
        batch["rms"][:1],
        batch["speaker"][:1],
        mask[:1],
        steps=2,
        guidance_strength=1.5,
    )
    assert generated.shape == (1, 7, channels)
    assert generated.isfinite().all()
