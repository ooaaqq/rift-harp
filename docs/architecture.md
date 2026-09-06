# Architecture

RIFT-HARP maps source content, pitch, energy, and a singer condition to a
128-bin target log-mel spectrogram:

$$
(C, F_0, R, e) \xrightarrow{\mathrm{HARP}} M.
$$

## Acoustic representation

The 44.1 kHz frontend uses a 2048-sample FFT and window, hop size 512, Slaney
mel filters from 40 Hz to 16 kHz, natural-log magnitude, and `center=false`.
The frozen flow transform centers the log-mel vector \(x\), rotates it with a
full PCA basis \(B\), and applies clipped partial whitening \(g\):

$$
y_1 = g \odot B(x-\mu).
$$

## Residual flow objective

For Gaussian noise \(\epsilon\) and timestep \(t\), training constructs

$$
y_t=(1-t)\epsilon+t y_1.
$$

Let \(\lambda\) be the persisted variance of each transformed mel mode:

$$
q=t^2\lambda+(1-t)^2,
\quad c_{\mathrm{in}}=q^{-1/2},
$$

$$
c_{\mathrm{skip}}=\frac{t\lambda-(1-t)}{q},
\quad c_{\mathrm{out}}=\sqrt{\frac{\lambda}{q}}.
$$

The network predicts the standardized residual target

$$
F^\star=\frac{(y_1-\epsilon)-c_{\mathrm{skip}}y_t}
{c_{\mathrm{out}}},
$$

with masked mean-squared error over valid frames and mel channels. Sampling
integrates the transformed state and applies the inverse transform once at the
endpoint.

## Network

The temporal backbone is 1024 channels wide with 16 blocks, 16 attention heads,
QK normalization, RoPE, and a 2816-channel gated convolutional FFN with
depthwise kernel 31. Separate content, pitch, harmonic, and energy stems are
mixed before the backbone. Harmonic adapters reinject pitch coordinates before
blocks 4, 8, and 12.

Each block uses low-rank multiplicative time-singer modulation:

$$
t_l=W_t(t),\quad s_l=W_s(e),
$$

$$
m=\operatorname{SiLU}(W_m[t_l,s_l,t_l\odot s_l]),
\quad \mathrm{mod}=W_o(m).
$$

The final head predicts all 128 transformed mel modes. PC-NSF reconstructs the
waveform from the predicted mel and the same final F0 track used by the acoustic
model.
