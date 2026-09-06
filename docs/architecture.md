# Architecture

RIFT-HARP maps source content, pitch, energy, and a singer condition to a
128-bin target log-mel spectrogram:

$$
M=f_\theta(C,F_0,R,e).
$$

Here $C$ is ContentVec, $F_0$ is frame-level pitch, $R$ is frame-level RMS,
and $e$ is the singer code.

## Acoustic representation

The 44.1 kHz frontend uses a 2048-sample FFT and window, hop size 512, Slaney
mel filters from 40 Hz to 16 kHz, natural-log magnitude, and `center=false`.
The frozen flow transform centers the log-mel vector $x$, rotates it with a
full PCA basis $B$, and applies clipped partial whitening $g$:

$$
y_1 = g \odot B(x-\mu).
$$

## Residual flow objective

For Gaussian noise $\epsilon$ and timestep $t$, training constructs

$$
y_t=(1-t)\epsilon+t y_1.
$$

Let $\lambda$ be the persisted variance of each transformed mel mode:

$$
q=t^2\lambda+(1-t)^2,
\quad c_{\mathrm{in}}=q^{-1/2},
$$

$$
c_{\mathrm{skip}}=\frac{t\lambda-(1-t)}{q},
\quad c_{\mathrm{out}}=\sqrt{\frac{\lambda}{q}}.
$$

The network receives the normalized state and predicts a standardized residual:

$$
\widehat F=f_\theta(c_{\mathrm{in}}y_t,C,F_0,R,e,t).
$$

Its target is

$$
F^\star=\frac{(y_1-\epsilon)-c_{\mathrm{skip}}y_t}
{c_{\mathrm{out}}}.
$$

For valid-frame mask $m_{it}$ and 128 mel modes, the loss is

$$
L=\frac{\sum_{i,t,k}m_{it}(\widehat F_{itk}-F^\star_{itk})^2}
{128\sum_{i,t}m_{it}}.
$$

Sampling integrates the transformed state and applies the inverse transform
once at the endpoint.

## Network

The temporal backbone is 1024 channels wide with 16 blocks, 16 attention heads,
QK normalization, RoPE, and a 2816-channel gated convolutional FFN with
depthwise kernel 31. Separate content, pitch, harmonic, and energy stems are
mixed before the backbone. Harmonic adapters reinject pitch coordinates before
blocks 4, 8, and 12.

Each block projects the time code $h_t$ and singer code $e$ into a shared
low-rank space:

$$
u_t=W_t(h_t),\qquad u_s=W_s(e),
$$

$$
m=\mathrm{SiLU}\left(W_m[u_t;u_s;u_t\odot u_s]\right),
\qquad a=W_o(m).
$$

The vector $a$ supplies the shift, scale, and gate values used by the block.

The final head predicts all 128 transformed mel modes. PC-NSF reconstructs the
waveform from the predicted mel and the same final F0 track used by the acoustic
model.
