import torch

from harp.flow_transform import FlowTransform, fit_flow_transform, orthonormal_dct


def test_dct_is_orthonormal() -> None:
    basis = orthonormal_dct(16)
    torch.testing.assert_close(basis @ basis.T, torch.eye(16), atol=2e-6, rtol=2e-6)


def test_transform_round_trip_and_artifact(tmp_path) -> None:
    channels = 8
    transform = FlowTransform(
        mean=torch.randn(channels),
        basis=orthonormal_dct(channels),
        gain=torch.linspace(0.5, 2, channels),
        lambda_raw=torch.ones(channels),
        lambda_effective=torch.ones(channels),
        metadata={"artifact_type": "flow_transform_v1"},
    )
    value = torch.randn(3, 11, channels)
    torch.testing.assert_close(
        transform.inverse(transform.transform(value)), value, atol=2e-6, rtol=2e-6
    )
    path = tmp_path / "flow_transform_v1.npz"
    digest = transform.save(path)
    loaded = FlowTransform.load(path)
    assert len(digest) == 64
    torch.testing.assert_close(loaded.basis, transform.basis)


def test_fit_uses_pca_when_dct_coordinates_remain_correlated() -> None:
    torch.manual_seed(10)
    source = torch.randn(20_000, 8) * torch.linspace(0.8, 1.2, 8)
    mixing, _ = torch.linalg.qr(torch.randn(8, 8))
    frames = source @ mixing.T + torch.linspace(-2, 1, 8)
    transform = fit_flow_transform(
        frames,
        seed=7,
        sampler_hash="sampler",
        dataset_manifest_hash="manifest",
        crop_policy_hash="crop",
    )
    assert transform.metadata["basis_kind"] == "pca"
    assert transform.metadata["validation"]["offdiag_ratio"] <= 0.10
    assert transform.metadata["validation"]["max_abs_corr"] <= 0.30
    assert torch.median(transform.lambda_raw).isclose(torch.tensor(1.0), atol=0.05)
